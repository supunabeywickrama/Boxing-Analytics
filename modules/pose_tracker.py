import cv2
import mediapipe as mp

class PoseTracker:
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(frame_rgb)
        return res

    @staticmethod
    def extract_keypoints(results, width, height):
        if results.pose_landmarks is None:
            return None
        pts = {}
        for i, lm in enumerate(results.pose_landmarks.landmark):
            pts[i] = (int(lm.x * width), int(lm.y * height))
        return pts
