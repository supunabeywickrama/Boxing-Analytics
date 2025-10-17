# modules/metrics.py
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
import math
import csv
import os

@dataclass
class SessionStatsRow:
    time_s: float
    total_punches: int
    jabs: int
    crosses: int
    hooks: int
    uppercuts: int
    energy_ppm: float
    avg_speed_norm: float
    avg_power_idx: float
    avg_tech_score: float
    fatigue_level: float

class EnergyTracker:
    """Rolling punches-per-minute (PPM)."""
    def __init__(self, window_s=60.0):
        self.window_s = window_s
        self.events = deque()  # store event times

    def update(self, t, new_events):
        # Trim old
        while self.events and (t - self.events[0]) > self.window_s:
            self.events.popleft()
        # Append new
        for _ in new_events:
            self.events.append(t)

    def punches_per_min(self):
        return len(self.events) * (60.0 / self.window_s)

class ComboTracker:
    """Simple combo buffer within window."""
    def __init__(self, window_s=1.2):
        self.window_s = window_s
        self.buffer = deque()

    def update(self, t, events):
        for e in events:
            self.buffer.append((t, e.punch_type))
        while self.buffer and (t - self.buffer[0][0]) > self.window_s:
            self.buffer.popleft()

    def current_combo(self):
        return "-".join([p for (_, p) in list(self.buffer)])

class FatigueModel:
    """
    Toy fatigue model:
      - Starts at 0 (fresh), rises with exertion ~ power + cadence
      - Decays exponentially over time
    Output is bounded [0..1] and can be used to scale visuals / alerts.
    """
    def __init__(self, rise_per_punch=0.02, rise_power_scale=0.02, decay_half_life_s=45.0):
        self.level = 0.0
        self.decay_lambda = math.log(2) / max(1e-6, decay_half_life_s)
        self.rise_per_punch = rise_per_punch
        self.rise_power_scale = rise_power_scale
        self.t_prev = None

    def update(self, t, events):
        if self.t_prev is None:
            self.t_prev = t
        dt = max(0.0, t - self.t_prev)
        # Exponential decay
        self.level *= math.exp(-self.decay_lambda * dt)
        # Add load from new punches
        for e in events:
            # base rise + scaled by power
            self.level += self.rise_per_punch + self.rise_power_scale * min(5.0, e.power_idx)
        self.level = max(0.0, min(1.0, self.level))
        self.t_prev = t
        return self.level

class RollingStats:
    """Keep a rolling window of scalar metrics for smoothing overlays."""
    def __init__(self, maxlen=150):  # ~5s at 30 FPS
        self.speed = deque(maxlen=maxlen)
        self.power = deque(maxlen=maxlen)
        self.tech  = deque(maxlen=maxlen)

    def push_events(self, events):
        for e in events:
            self.speed.append(e.speed_norm)
            self.power.append(e.power_idx)
            self.tech.append(e.tech_score)

    def avg(self, q):
        arr = getattr(self, q)
        return sum(arr)/len(arr) if arr else 0.0

class SessionAggregator:
    """
    Aggregates session metrics and exports a per-video summary CSV.
    Call .step(t, events, energy_ppm, fatigue, rolling_stats) each frame.
    """
    def __init__(self):
        self.t_last_row = 0.0
        self.total_punches = 0
        self.by_class = defaultdict(int)
        self.rows = []
        self.speeds = []
        self.powers = []
        self.techs  = []

    def step(self, t, events, energy_ppm, fatigue, rolling_stats, row_every_s=1.0):
        # Accumulate per-event
        for e in events:
            self.total_punches += 1
            self.by_class[e.punch_type] += 1
            self.speeds.append(e.speed_norm)
            self.powers.append(e.power_idx)
            self.techs.append(e.tech_score)

        # Store one row per second for a session timeline
        if (t - self.t_last_row) >= row_every_s:
            row = SessionStatsRow(
                time_s=round(t, 2),
                total_punches=self.total_punches,
                jabs=self.by_class.get("Jab", 0),
                crosses=self.by_class.get("Cross", 0),
                hooks=self.by_class.get("Hook", 0),
                uppercuts=self.by_class.get("Uppercut", 0),
                energy_ppm=round(energy_ppm, 2),
                avg_speed_norm=round(rolling_stats.avg("speed"), 3),
                avg_power_idx=round(rolling_stats.avg("power"), 3),
                avg_tech_score=round(rolling_stats.avg("tech"), 3),
                fatigue_level=round(fatigue, 3),
            )
            self.rows.append(row)
            self.t_last_row = t

    def export_timeline_csv(self, out_csv_path):
        if not self.rows:
            return
        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
        with open(out_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(self.rows[0]).keys()))
            w.writeheader()
            for r in self.rows:
                w.writerow(asdict(r))

    def export_session_summary(self, out_csv_path, video_name):
        # Final aggregates
        punches = self.total_punches
        j = self.by_class.get("Jab", 0)
        c = self.by_class.get("Cross", 0)
        h = self.by_class.get("Hook", 0)
        u = self.by_class.get("Uppercut", 0)
        avg_speed = sum(self.speeds)/len(self.speeds) if self.speeds else 0.0
        avg_power = sum(self.powers)/len(self.powers) if self.powers else 0.0
        avg_tech  = sum(self.techs)/len(self.techs)   if self.techs  else 0.0

        header = ["video","total","jab","cross","hook","uppercut","avg_speed_norm","avg_power_idx","avg_tech_score"]
        write_header = not os.path.exists(out_csv_path)
        with open(out_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow([video_name, punches, j, c, h, u, round(avg_speed,3), round(avg_power,3), round(avg_tech,3)])
