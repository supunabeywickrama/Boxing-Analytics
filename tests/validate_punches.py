# tests/validate_punches.py
import os, glob, csv
from modules.punch_classifier import PunchClassifier, PunchThresholds
from modules.pose_tracker import PoseTracker
from modules.yolo_detector import YOLODetector
import cv2

INPUT = "samples"
OUT_DIR = "outputs"
USE_YOLO = True
SPEED_PEAK = 0.80  # try 0.6–1.0
MIN_INTERVAL = 0.25

def list_videos(folder):
    vids = []
    for ext in ("*.mp4","*.mov","*.avi","*.mkv"):
        vids += glob.glob(os.path.join(folder, ext))
    return vids

def validate_one(video_path):
    os.makedirs(OUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] cannot open:", video_path); return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    th = PunchThresholds(speed_peak=SPEED_PEAK, min_interval_s=MIN_INTERVAL)
    clf = PunchClassifier(dt_s=1.0/max(1.0,fps), thresholds=th)
    pose = PoseTracker()
    yolo = YOLODetector() if USE_YOLO else None

    counts = {"Jab":0,"Cross":0,"Hook":0,"Uppercut":0}
    total_frames, pose_ok = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        total_frames += 1

        if yolo:
            bbox = yolo.detect_person(frame)
            if bbox:
                x1,y1,x2,y2 = bbox
                pad=10
                x1=max(0,x1-pad); y1=max(0,y1-pad)
                x2=min(W-1,x2+pad); y2=min(H-1,y2+pad)
                roi = frame[y1:y2, x1:x2].copy()
                res = pose.process(roi)
                pts = pose.extract_keypoints(res, roi.shape[1], roi.shape[0])
                if pts: pts = {k:(v[0]+x1, v[1]+y1) for k,v in pts.items()}
            else:
                res = pose.process(frame)
                pts = pose.extract_keypoints(res, W, H)
        else:
            res = pose.process(frame)
            pts = pose.extract_keypoints(res, W, H)

        if pts: pose_ok += 1
        events, _ = clf.update(total_frames/fps, pts)
        for e in (events or []):
            counts[e.punch_type] += 1

    cap.release()
    coverage = 100.0 * pose_ok / max(1,total_frames)
    return {
        "video": os.path.basename(video_path),
        "pose_coverage_%": round(coverage,1),
        **counts
    }

def main():
    videos = list_videos(INPUT)
    if not videos:
        print("[WARN] no videos in samples/")
        return

    rows = []
    for vp in videos:
        res = validate_one(vp)
        if res:
            rows.append(res)
            print(res)

    # global summary
    total = {"Jab":0,"Cross":0,"Hook":0,"Uppercut":0}
    for r in rows:
        for k in total:
            total[k] += r[k]
    print("\n=== GLOBAL SUMMARY ===")
    print(total)

    # save csv
    out_csv = os.path.join(OUT_DIR, "validation_summary.csv")
    with open(out_csv,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("[OK] saved", out_csv)

if __name__ == "__main__":
    main()
