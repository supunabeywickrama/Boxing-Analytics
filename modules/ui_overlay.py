import cv2, numpy as np, math, time
from .punch_classifier import LEFT_SHOULDER, RIGHT_SHOULDER
from .effects import draw_rounded_rect, put_glow_text, draw_vu_meter

PRIMARY = (0, 200, 255)   # cyan-yellow neon
ACCENT  = (120, 120, 255) # pinkish
TEXT    = (245, 245, 245)

def draw_pose(frame, pts, color=(0,255,180)):
    if pts is None: return frame
    for (_, (x,y)) in pts.items():
        cv2.circle(frame, (int(x),int(y)), 3, color, -1)
    return frame

def draw_bars(frame, x, y, w, h, label, value, max_value=1.0):
    v = float(np.clip(value / max_value, 0, 1))
    bg = (25, 25, 30)
    draw_rounded_rect(frame, (x, y, w, h), 8, bg, -1)
    inner = int((w-4) * v)
    cv2.rectangle(frame, (x+2, y+2), (x+2+inner, y+h-2), PRIMARY, -1)
    cv2.putText(frame, f"{label}: {value:.2f}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT, 1, cv2.LINE_AA)

def put_panel(frame, lines, x=20, y=30, line_h=26):
    pad_x, pad_y = 14, 14
    w = max([cv2.getTextSize(L, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] for L in lines] + [180]) + pad_x*2
    h = line_h * len(lines) + pad_y
    draw_rounded_rect(frame, (x-10, y-24, w, h), 10, (30,30,40), -1)
    for i, txt in enumerate(lines):
        cv2.putText(frame, txt, (x, y + i*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT, 2, cv2.LINE_AA)

def face_direction_hint(pts):
    try:
        ls = np.array(pts[LEFT_SHOULDER]); rs = np.array(pts[RIGHT_SHOULDER])
        return "Facing Right" if (rs[0]-ls[0]) > 0 else "Facing Left"
    except Exception:
        return "Unknown"

def punch_banner(frame, text, W, H, t_norm):
    # t_norm in [0..1]: anim progress
    t_norm = np.clip(t_norm, 0, 1)
    cx, cy = W//2, int(0.18*H)
    s = 1.0 + 0.2*np.sin(t_norm*math.pi)  # subtle scale pop
    color = (0, 255, 255) if text in ("Jab","Cross") else (255, 120, 200)
    put_glow_text(frame, text, (cx - 120, cy), scale=1.6*s, color=(250,250,250), glow=color)
