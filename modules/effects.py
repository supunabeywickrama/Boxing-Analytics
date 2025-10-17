# modules/effects.py
import cv2, numpy as np, random, math, time
from collections import deque

def _clamp01(x): return max(0.0, min(1.0, x))

def draw_rounded_rect(img, rect, radius, color, thickness=-1):
    x, y, w, h = rect
    overlay = img.copy()
    cv2.rectangle(overlay, (x+radius, y), (x+w-radius, y+h), color, thickness)
    cv2.rectangle(overlay, (x, y+radius), (x+w, y+h-radius), color, thickness)
    for (cx, cy) in [(x+radius, y+radius), (x+w-radius, y+radius), (x+radius, y+h-radius), (x+w-radius, y+h-radius)]:
      cv2.circle(overlay, (cx, cy), radius, color, thickness)
    alpha = 1.0 if thickness != -1 else 1.0
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def put_glow_text(img, text, org, font=cv2.FONT_HERSHEY_DUPLEX, scale=1.2, color=(255,255,255), glow=(10, 255, 200), thickness=2):
    # blur-based glow
    (x,y) = org
    base = img.copy()
    cv2.putText(base, text, (x,y), font, scale, glow, 6, cv2.LINE_AA)
    glow1 = cv2.GaussianBlur(base, (0,0), 4)
    cv2.addWeighted(glow1, 0.6, img, 1.0, 0, img)
    cv2.putText(img, text, (x,y), font, scale, color, thickness, cv2.LINE_AA)

class Particle:
    __slots__ = ("x","y","vx","vy","life","maxlife","color")
    def __init__(self, x, y, speed, dir_rad, life, color):
        self.x, self.y = x, y
        self.vx = speed * math.cos(dir_rad)
        self.vy = speed * math.sin(dir_rad)
        self.life = self.maxlife = life
        self.color = color

    def step(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vx *= 0.98
        self.vy *= 0.98
        self.life -= dt

    def alpha(self):
        return _clamp01(self.life / self.maxlife)

class ParticleSystem:
    def __init__(self, max_count=200):
        self.particles = deque(maxlen=max_count)

    def burst(self, x, y, n=40, speed=300, life=0.6, hue="auto"):
        for i in range(n):
            angle = random.uniform(0, 2*math.pi)
            sp = random.uniform(0.4, 1.0) * speed
            c = (0, 255, 255) if hue=="auto" else hue
            self.particles.append(Particle(x, y, sp, angle, random.uniform(0.4,1.0)*life, c))

    def draw(self, img, dt):
        H, W = img.shape[:2]
        for p in list(self.particles):
            p.step(dt)
            if p.life <= 0: 
                self.particles.remove(p); 
                continue
            a = p.alpha()
            if 0 <= int(p.x) < W and 0 <= int(p.y) < H:
                cv2.circle(img, (int(p.x), int(p.y)), 2, (int(p.color[0]*a), int(p.color[1]*a), int(p.color[2]*a)), -1)

class Trail:
    def __init__(self, maxlen=20, color=(0,255,180)):
        self.pts = deque(maxlen=maxlen)
        self.color = color
    def add(self, x, y): self.pts.append((x,y))
    def draw(self, img):
        for i in range(1, len(self.pts)):
            x1,y1 = self.pts[i-1]; x2,y2 = self.pts[i]
            a = i/len(self.pts)
            c = (int(self.color[0]*(a)), int(self.color[1]*(a)), int(self.color[2]*(a)))
            cv2.line(img, (int(x1),int(y1)), (int(x2),int(y2)), c, 2)

def draw_vu_meter(img, x, y, w, h, value, label="Energy", bars=16):
    # animated vertical bars
    value = _clamp01(value)
    bw = w // bars
    for i in range(bars):
        t = (i+1)/bars
        on = t <= value
        col = (0, 220, 255) if on else (50, 50, 60)
        x1 = x + i*bw + 2
        y1 = y
        cv2.rectangle(img, (x1, y1), (x1 + bw - 4, y + h), col, -1)
    cv2.putText(img, f"{label}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,230,230), 1, cv2.LINE_AA)
