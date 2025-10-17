import argparse, os, glob, csv
import cv2
from tqdm import tqdm

from modules.pose_tracker import PoseTracker
from modules.yolo_detector import YOLODetector
from modules.effects import ParticleSystem, Trail, draw_vu_meter
from modules.punch_classifier import PunchClassifier, PunchThresholds
from modules.metrics import (
    EnergyTracker, ComboTracker, FatigueModel, RollingStats,
    SessionAggregator
)
from modules.ui_overlay import draw_pose, draw_bars, put_panel, face_direction_hint, punch_banner


def process_video(in_path, out_dir, use_yolo=True, thresholds: PunchThresholds = None):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {in_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = os.path.join(out_dir, os.path.splitext(os.path.basename(in_path))[0] + "_annotated.mp4")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (W, H))

    # --- FX systems ---
    particles = ParticleSystem(max_count=400)
    left_trail  = Trail(maxlen=22, color=(0,255,180))
    right_trail = Trail(maxlen=22, color=(255,180,60))
    last_banner_ts = -1.0
    last_banner_txt = "—"

    # --- Core CV/ML ---
    pose = PoseTracker(model_complexity=1)
    yolo = YOLODetector() if use_yolo else None
    th = thresholds or PunchThresholds()
    classifier = PunchClassifier(dt_s=1.0 / max(1.0, fps), thresholds=th)

    # --- Metrics ---
    energy = EnergyTracker(window_s=60.0)
    combo = ComboTracker(window_s=th.combo_window_s)
    fatigue = FatigueModel(rise_per_punch=0.02, rise_power_scale=0.02, decay_half_life_s=45.0)
    roll = RollingStats(maxlen=int(5 * fps))  # 5-second smoothing
    session = SessionAggregator()

    # per-punch CSV
    out_csv = os.path.join(out_dir, os.path.splitext(os.path.basename(in_path))[0] + "_punches.csv")
    csv_f = open(out_csv, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["time_s","hand","punch_type","speed_norm","power_idx","tech_score"])

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = tqdm(total=total if total>0 else None, desc=f"Processing {os.path.basename(in_path)}")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / fps

            # YOLO ROI (optional)
            if yolo:
                bbox = yolo.detect_person(frame)
                if bbox is not None:
                    x1,y1,x2,y2 = bbox
                    pad = 10
                    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
                    x2 = min(W-1, x2 + pad); y2 = min(H-1, y2 + pad)
                    roi = frame[y1:y2, x1:x2].copy()
                    res = pose.process(roi)
                    pts = pose.extract_keypoints(res, roi.shape[1], roi.shape[0])
                    if pts is not None:
                        pts = {k:(v[0]+x1, v[1]+y1) for k,v in pts.items()}
                else:
                    res = pose.process(frame)
                    pts = pose.extract_keypoints(res, W, H)
            else:
                res = pose.process(frame)
                pts = pose.extract_keypoints(res, W, H)

            # Trails on wrists (if present)
            if pts is not None:
                if 15 in pts:
                    lx, ly = pts[15]; left_trail.add(lx, ly)
                if 16 in pts:
                    rx, ry = pts[16]; right_trail.add(rx, ry)

            # Classify punches
            events, _ = classifier.update(t, pts)
            events = events or []

            # Metrics update
            energy.update(t, events)
            combo.update(t, events)
            f_level = fatigue.update(t, events)
            roll.push_events(events)

            # Log per-punch
            for e in events:
                csv_w.writerow([f"{e.t:.3f}", e.hand, e.punch_type, f"{e.speed_norm:.3f}", f"{e.power_idx:.3f}", f"{e.tech_score:.3f}"])

            # Per-second timeline aggregation
            session.step(
                t=t,
                events=events,
                energy_ppm=energy.punches_per_min(),
                fatigue=f_level,
                rolling_stats=roll,
                row_every_s=1.0
            )

            # FX triggers on punch
            for e in events:
                if e.hand == "Left" and pts is not None and 15 in pts:
                    x,y = pts[15]
                elif e.hand == "Right" and pts is not None and 16 in pts:
                    x,y = pts[16]
                else:
                    x,y = int(0.5*W), int(0.5*H)
                particles.burst(x, y, n=48, speed=360, life=0.5, hue=(0,255,255))
                last_banner_ts = t
                last_banner_txt = e.punch_type

            # ===== Overlay =====
            # Trails under pose
            left_trail.draw(frame)
            right_trail.draw(frame)

            # Pose
            draw_pose(frame, pts)

            # Current burst (this frame)
            speed_bar = max([e.speed_norm for e in events], default=0.0)
            power_bar = max([e.power_idx   for e in events], default=0.0)

            # Rolling averages (5s window)
            avg_speed = roll.avg("speed")
            avg_power = roll.avg("power")
            avg_tech  = roll.avg("tech")

            # Left HUD bars
            draw_bars(frame, 20, H-120, 220, 18, "Speed (curr)", speed_bar, max_value=3.0)
            draw_bars(frame, 20, H-90,  220, 18, "Power (curr)", power_bar, max_value=5.0)
            draw_bars(frame, 20, H-60,  220, 18, "Speed (avg)",  avg_speed, max_value=3.0)

            # Right VU meter (Energy)
            energy_norm = min(1.0, energy.punches_per_min()/200.0)
            draw_vu_meter(frame, W-220, H-120, 200, 90, energy_norm, label="Energy")

            # Top-left panel
            last_label = events[-1].punch_type if events else "—"
            lines = [
                f"Punch: {last_label}",
                f"Combo: {combo.current_combo()}",
                f"Facing: {face_direction_hint(pts)}",
                f"Avg Tech: {avg_tech:.2f} | Fatigue: {f_level:.2f}",
            ]
            put_panel(frame, lines, x=20, y=30)

            # Particles (dt in seconds)
            particles.draw(frame, dt=1.0/max(1.0, fps))

            # Punch banner (animate ~0.6s after last event)
            if last_banner_ts >= 0 and (t - last_banner_ts) < 0.6:
                t_norm = (t - last_banner_ts)/0.6
                punch_banner(frame, last_banner_txt, W, H, t_norm)

            writer.write(frame)
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        writer.release()
        csv_f.close()

    # Export timelines + session summary
    timeline_csv = os.path.join(out_dir, os.path.splitext(os.path.basename(in_path))[0] + "_timeline.csv")
    session.export_timeline_csv(timeline_csv)

    summary_csv = os.path.join(out_dir, "session_summary.csv")
    session.export_session_summary(summary_csv, video_name=os.path.basename(in_path))

    print(f"[OK] Saved video: {out_video}")
    print(f"[OK] Punches CSV: {out_csv}")
    print(f"[OK] Timeline CSV: {timeline_csv}")
    print(f"[OK] Appended summary: {summary_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to video file or a folder of videos")
    ap.add_argument("--output_dir", default="outputs", help="Where to save annotated videos and CSV")
    ap.add_argument("--no-yolo", action="store_true", help="Disable YOLO, use pose-only")

    # Threshold tuning
    ap.add_argument("--speed-peak", type=float, default=0.80)
    ap.add_argument("--min-interval", type=float, default=0.25)
    ap.add_argument("--forward-proj", type=float, default=0.50)
    ap.add_argument("--up-ratio", type=float, default=0.60)
    ap.add_argument("--elbow-mid", type=float, default=120.0)
    ap.add_argument("--elbow-tol", type=float, default=30.0)
    ap.add_argument("--combo-window", type=float, default=1.2)

    args = ap.parse_args()

    th = PunchThresholds(
        speed_peak=args.speed_peak,
        min_interval_s=args.min_interval,
        forward_proj_ratio=args.forward_proj,
        up_ratio=args.up_ratio,
        elbow_bent_mid=args.elbow_mid,
        elbow_bent_tol=args.elbow_tol,
    )
    th.combo_window_s = args.combo_window

    if os.path.isdir(args.input):
        vids = []
        for ext in ("*.mp4","*.mov","*.avi","*.mkv"):
            vids += glob.glob(os.path.join(args.input, ext))
        if not vids:
            print("[WARN] No videos found in folder.")
            return
        for vp in vids:
            process_video(vp, args.output_dir, use_yolo=not args.no_yolo, thresholds=th)
    else:
        process_video(args.input, args.output_dir, use_yolo=not args.no_yolo, thresholds=th)

if __name__ == "__main__":
    main()
