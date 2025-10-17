# modules/punch_classifier.py
from collections import deque
import numpy as np
import math
from dataclasses import dataclass

# MediaPipe indices
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

@dataclass
class PunchThresholds:
    # core
    speed_peak: float = 0.80     # normalized wrist speed peak (vs torso size per second)
    min_interval_s: float = 0.25  # refractory time between same-hand punches
    # motion decomposition
    up_ratio: float = 0.60        # how much upward vs lateral for uppercut
    forward_proj_ratio: float = 0.50  # how much projected forward for straights
    # elbow geometry
    elbow_straight_min: float = 155.0  # considered "straight"
    elbow_bent_mid: float = 120.0      # ideal bend for hooks/uppercuts
    elbow_bent_tol: float = 30.0       # tolerance around ideal
    # smoothing
    ema_pos_alpha: float = 0.35        # EMA for wrist position
    ema_speed_alpha: float = 0.45      # EMA for speed magnitude
    # noise guard
    min_move_px_norm: float = 0.05     # ignore tiny movements
    # combo
    combo_window_s: float = 1.2

@dataclass
class PunchEvent:
    t: float
    hand: str
    punch_type: str
    speed_norm: float
    power_idx: float
    tech_score: float

class EMA:
    def __init__(self, alpha):
        self.alpha = alpha
        self.v = None
    def update(self, x):
        if self.v is None:
            self.v = np.array(x, dtype=float)
        else:
            self.v = self.alpha * np.array(x, float) + (1 - self.alpha) * self.v
        return self.v

class PunchClassifier:
    def __init__(self, dt_s=1/30.0, thresholds: PunchThresholds = None):
        self.dt_s = dt_s
        self.th = thresholds or PunchThresholds()
        self.hist = {"L": deque(maxlen=12), "R": deque(maxlen=12)}
        self.last_peak = {"L": -1e9, "R": -1e9}
        self.pos_ema = {"L": EMA(self.th.ema_pos_alpha), "R": EMA(self.th.ema_pos_alpha)}
        self.speed_ema = {"L": EMA(self.th.ema_speed_alpha), "R": EMA(self.th.ema_speed_alpha)}

    @staticmethod
    def torso_size(pts):
        try:
            ls, rs = np.array(pts[LEFT_SHOULDER]), np.array(pts[RIGHT_SHOULDER])
            lh, rh = np.array(pts[LEFT_HIP]), np.array(pts[RIGHT_HIP])
            w = np.linalg.norm(ls - rs)
            h = np.linalg.norm(lh - rh)
            return max(10.0, (w + h) / 2.0)
        except Exception:
            return 100.0

    @staticmethod
    def angle(a, b, c):
        a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
        ba = a - b
        bc = c - b
        den = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
        cosang = np.clip(np.dot(ba, bc) / den, -1.0, 1.0)
        return math.degrees(math.acos(cosang))

    @staticmethod
    def facing_right(pts):
        # Shoulder line vector; if right shoulder is to the right of left shoulder → facing right
        try:
            return (pts[RIGHT_SHOULDER][0] - pts[LEFT_SHOULDER][0]) > 0
        except Exception:
            return True

    def _classify_from_motion(self, hand_code, vx, vy, forearm_angle, forward_dir, speed_norm):
        # Decompose velocity
        vel_vec = np.array([vx, vy], float)
        mag = np.linalg.norm(vel_vec) + 1e-6
        proj_forward = np.dot(vel_vec / mag, forward_dir)  # [-1,1] w.r.t. shoulder line
        forward = proj_forward > self.th.forward_proj_ratio
        upward = (-vy) > self.th.up_ratio * (abs(vx) + 1e-6)  # y+ is down

        elbow_straight = forearm_angle >= self.th.elbow_straight_min
        elbow_ideal_hook = abs(self.th.elbow_bent_mid - forearm_angle) <= self.th.elbow_bent_tol

        if upward and not elbow_straight:
            return "Uppercut"
        if not forward and elbow_ideal_hook:
            return "Hook"
        # default straight: lead = Jab, rear = Cross
        return "Jab" if hand_code == "L" else "Cross"

    def update(self, t, pts):
        if pts is None:
            return [], None

        scale = self.torso_size(pts)
        events = []
        face_right = self.facing_right(pts)

        hands = [
            ("L", LEFT_WRIST, LEFT_ELBOW, LEFT_SHOULDER),
            ("R", RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER),
        ]

        # Forward direction = shoulder line (left->right) normalized
        try:
            shoulder_line = np.array(pts[RIGHT_SHOULDER]) - np.array(pts[LEFT_SHOULDER])
            forward_dir = shoulder_line / (np.linalg.norm(shoulder_line) + 1e-6)
        except Exception:
            forward_dir = np.array([1.0, 0.0])

        for hand_code, WR, EL, SH in hands:
            if WR not in pts or EL not in pts or SH not in pts:
                continue

            # EMA-smoothed position
            wpos = self.pos_ema[hand_code].update(pts[WR])
            self.hist[hand_code].append((t, wpos[0], wpos[1]))

            if len(self.hist[hand_code]) < 2:
                continue

            (t1, x1, y1) = self.hist[hand_code][-2]
            (t2, x2, y2) = self.hist[hand_code][-1]
            dt = max(1e-3, t2 - t1)
            vx = (x2 - x1) / dt
            vy = (y2 - y1) / dt

            # Normalize speed by torso size (per-second) and smooth
            speed_norm_raw = (np.hypot(vx, vy) / max(1.0, scale))
            speed_norm = float(self.speed_ema[hand_code].update([speed_norm_raw])[0])

            # ignore jitter
            if speed_norm < self.th.min_move_px_norm:
                continue

            # peak detection + refractory
            if speed_norm > self.th.speed_peak and (t2 - self.last_peak[hand_code]) >= self.th.min_interval_s:
                self.last_peak[hand_code] = t2

                wr = np.array(wpos)
                el = np.array(pts[EL])
                sh = np.array(pts[SH])
                forearm_angle = self.angle(sh, el, wr)

                # Map “left/right hand code” to lead/rear depending on stance
                # If facing right: left = lead → jab; right = rear → cross (for straights)
                ptype = self._classify_from_motion(hand_code, vx, vy, forearm_angle, forward_dir, speed_norm)

                # Power ~ speed^2 (bounded)
                power_idx = float(np.clip(speed_norm ** 2, 0, 5.0))

                # Technique: straight prefers straighter elbow; hook/upper prefers mid bend
                if ptype in ("Jab", "Cross"):
                    tech = 1.0 - abs(180.0 - forearm_angle) / 90.0
                else:
                    tech = 1.0 - abs(self.th.elbow_bent_mid - forearm_angle) / self.th.elbow_bent_tol
                tech = float(np.clip(tech, 0.0, 1.0))

                events.append(
                    PunchEvent(
                        t=t2,
                        hand=("Left" if hand_code == "L" else "Right"),
                        punch_type=ptype,
                        speed_norm=float(speed_norm),
                        power_idx=power_idx,
                        tech_score=tech,
                    )
                )
        return events, scale
