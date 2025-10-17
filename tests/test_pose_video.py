# tests/test_pose_video.py
import os, cv2, numpy as np
from modules.pose_tracker import PoseTracker

INPUT = "samples"  # file or folder
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def iter_videos(path):
    if os.path.isdir(path):
        exts = (".mp4",".mov",".avi",".mkv")
        for f in os.listdir(path):
            if f.lower().endswith(exts):
                yield os.path.join(path, f)
    else:
        yield path

def process_one(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open:", video_path); return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(video_path))[0] + "_pose_smoketest.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W,H))

    pose = PoseTracker(model_complexity=1)
    total, with_landmarks = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok: break
        total += 1

        res = pose.process(frame)
        pts = pose.extract_keypoints(res, W, H)
        if pts is not None:
            with_landmarks += 1
            for _, (x,y) in pts.items():
                cv2.circle(frame, (int(x),int(y)), 2, (0,255,0), -1)

        cv2.putText(frame, f"Pose frames: {with_landmarks}/{total}", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        writer.write(frame)

    cap.release(); writer.release()
    coverage = (with_landmarks/total*100) if total else 0
    print(f"[OK] {video_path} -> {out_path} | Pose coverage: {coverage:.1f}%")

def main():
    for vp in iter_videos(INPUT):
        process_one(vp)

if __name__ == "__main__":
    main()
