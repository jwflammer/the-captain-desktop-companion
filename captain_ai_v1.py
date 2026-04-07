# captain_ai.py
# © 2026 Proficient PC. All rights reserved.
# Requires: PyQt6, numpy, opencv-python, pyaudio

import sys, time, threading, random, math, sqlite3, os, json, shutil, uuid
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QLabel, QTextEdit, QFrame, QSizePolicy
)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QTextCursor, QCursor, QPen, QRadialGradient
from PyQt6.QtCore import Qt, QTimer, QPoint, QRectF


APP_VERSION = "Captain Prototype v1-MultiEye | Shell Phase 3 (Cyber-Core)"
SCHEMA_VERSION = 101
IDENTITY_VERSION = 1
CORE_SPEC = {
    "version": "core_v1_multi_eye",
    "schema_version": SCHEMA_VERSION,
    "identity_version": IDENTITY_VERSION,
    "visual_neurons": 4096,
    "audio_neurons": 1024,
    "touch_neurons": 256,
    "associative_neurons": 4096,
    "vocal_neurons": 1024,
    "compact_fovea_res": [64, 64],
    "compact_periphery_res": [96, 54],
    "audio_feature_bands": 512,
    "audio_display_bands": 64,
    "audio_sample_rate": 48000,
    "audio_fft_size": 4096,
}


# ==========================================
# DREAM STATE / SIM WORLD
# ==========================================
class SimWorld:
    def __init__(self):
        self.w, self.h = 1024, 1024
        self.x, self.y, self.theta = 512.0, 512.0, 0.0
        self.apples_eaten = 0
        self.treats = []
        self.obstacles = []
        self.laser_dist = 80.0
        self.left_whisker = 0.0
        self.right_whisker = 0.0
        self.bump = 0.0
        self.imu_tilt = 0.0
        self.charger = {"x": 120.0, "y": 904.0, "r": 60}
        self.fov = math.radians(60)
        self.rays = 128
        self.max_depth = 400.0
        self.social_targets = []
        self.light_phase = 0.0
        self._generate_obstacles()
        self._spawn_treats(5)
        self._spawn_social_targets(2)

    def _is_inside_obstacle(self, px, py, padding=0):
        for obs in self.obstacles:
            if (obs["x"] - padding < px < obs["x"] + obs["w"] + padding and
                obs["y"] - padding < py < obs["y"] + obs["h"] + padding):
                return True
        return False

    def _generate_obstacles(self):
        self.obstacles = []
        while len(self.obstacles) < 8:
            ow = random.randint(50, 200)
            oh = random.randint(50, 200)
            ox = random.randint(50, self.w - 250)
            oy = random.randint(50, self.h - 250)
            if (ox - 40 < 512.0 < ox + ow + 40) and (oy - 40 < 512.0 < oy + oh + 40):
                continue
            if (ox - 40 < self.charger["x"] < ox + ow + 40) and (oy - 40 < self.charger["y"] < oy + oh + 40):
                continue
            self.obstacles.append({"x": ox, "y": oy, "w": ow, "h": oh})

    def _spawn_treats(self, count):
        for _ in range(count):
            while True:
                tx = random.randint(50, self.w - 50)
                ty = random.randint(50, self.h - 50)
                if not self._is_inside_obstacle(tx, ty, padding=15) and math.hypot(tx - self.charger["x"], ty - self.charger["y"]) > 80:
                    self.treats.append({"x": tx, "y": ty})
                    break

    def _spawn_social_targets(self, count):
        self.social_targets = []
        for _ in range(count):
            while True:
                tx = random.randint(120, self.w - 120)
                ty = random.randint(120, self.h - 120)
                if not self._is_inside_obstacle(tx, ty, padding=40):
                    self.social_targets.append({
                        "x": float(tx),
                        "y": float(ty),
                        "theta": random.uniform(0, 360),
                        "speed": random.uniform(8.0, 18.0),
                        "radius": 22.0,
                    })
                    break

    def _update_social_targets(self, dt):
        self.light_phase += dt * 0.35
        for tgt in self.social_targets:
            if random.random() < 0.03:
                tgt["theta"] += random.uniform(-65, 65)
            rad = math.radians(tgt["theta"])
            nx = tgt["x"] + math.cos(rad) * tgt["speed"] * dt
            ny = tgt["y"] + math.sin(rad) * tgt["speed"] * dt
            if nx < 80 or nx > self.w - 80 or ny < 80 or ny > self.h - 80 or self._is_inside_obstacle(nx, ny, padding=28):
                tgt["theta"] += random.uniform(110, 250)
            else:
                tgt["x"], tgt["y"] = nx, ny

    def get_social_semantics(self):
        rad_heading = math.radians(self.theta)
        best = None
        for tgt in self.social_targets:
            dx = tgt["x"] - self.x
            dy = tgt["y"] - self.y
            dist = math.hypot(dx, dy)
            if dist > self.max_depth:
                continue
            ang = math.atan2(dy, dx)
            rel = (ang - rad_heading + math.pi) % (2 * math.pi) - math.pi
            if abs(rel) <= self.fov / 2.0:
                score = (1.0 / max(1.0, dist)) * (1.0 - abs(rel) / (self.fov / 2.0 + 1e-6))
                if best is None or score > best[0]:
                    best = (score, tgt, dist, rel)
        if best is None:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _, tgt, dist, rel = best
        face_present = 0.9
        face_x = float(np.clip(rel / (self.fov / 2.0), -1.0, 1.0))
        face_y = 0.0
        face_size = float(np.clip(160.0 / max(dist, 1.0), 0.0, 0.35))
        return np.array([face_present, face_x, face_y, face_size], dtype=np.float32)

    def _ray_distance(self, ang_deg, max_d=80, step=2):
        rad = math.radians(self.theta + ang_deg)
        for d in range(0, max_d, step):
            rx = self.x + 30 * math.cos(rad) + d * math.cos(rad)
            ry = self.y + 30 * math.sin(rad) + d * math.sin(rad)
            if self._is_inside_obstacle(rx, ry) or rx < 0 or rx > self.w or ry < 0 or ry > self.h:
                return float(d)
        return float(max_d)

    def step(self, v, w_rot, dt):
        rad = math.radians(self.theta)
        next_x = self.x + v * math.cos(rad) * dt
        next_y = self.y + v * math.sin(rad) * dt
        self.theta = (self.theta + w_rot * dt) % 360.0

        hit_wall = False
        proximity_warning = False

        if next_x < 15 or next_x > self.w - 15:
            hit_wall = True
            next_x = self.x
        if next_y < 15 or next_y > self.h - 15:
            hit_wall = True
            next_y = self.y

        if self._is_inside_obstacle(next_x, next_y, padding=15):
            hit_wall = True
            next_x = self.x
            next_y = self.y

        if not hit_wall and self._is_inside_obstacle(next_x, next_y, padding=40):
            proximity_warning = True

        self.x, self.y = next_x, next_y
        self._update_social_targets(dt)

        self.laser_dist = self._ray_distance(0)
        l = self._ray_distance(-35)
        r = self._ray_distance(35)
        self.left_whisker = 1.0 - min(1.0, l / 60.0)
        self.right_whisker = 1.0 - min(1.0, r / 60.0)
        self.bump = 1.0 if hit_wall else 0.0
        self.imu_tilt = float(np.clip(abs(w_rot) / 120.0, 0.0, 1.0))

        eaten = False
        for t in self.treats[:]:
            if math.hypot(self.x - t["x"], self.y - t["y"]) < 30:
                self.treats.remove(t)
                self._spawn_treats(1)
                self.apples_eaten += 1
                eaten = True
                if self.apples_eaten % 25 == 0:
                    self.x, self.y, self.theta = 512.0, 512.0, 0.0
                    self._generate_obstacles()
                    self.treats = []
                    self._spawn_treats(5)

        on_charger = math.hypot(self.x - self.charger["x"], self.y - self.charger["y"]) < self.charger["r"]
        return eaten, hit_wall, proximity_warning, on_charger

    def get_touch_sensors(self):
        return np.array([
            self.left_whisker,
            self.right_whisker,
            min(1.0, self.laser_dist / 80.0),
            self.bump,
            self.imu_tilt,
            1.0 if math.hypot(self.x - self.charger["x"], self.y - self.charger["y"]) < self.charger["r"] else 0.0
        ], dtype=np.float32)

    def get_ego_vision(self):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        ambient = 18 + int(12 * math.sin(self.light_phase))
        frame[:64, :, :] = max(0, ambient - 6)
        if math.hypot(self.x - self.charger["x"], self.y - self.charger["y"]) < self.charger["r"]:
            frame[64:, :, 0] = 50
            frame[64:, :, 2] = 50
        else:
            frame[64:, :, 1] = 20 + ambient // 4

        rad_heading = math.radians(self.theta)
        for i in range(self.rays):
            ray_angle = (rad_heading - self.fov / 2.0) + (float(i) / self.rays) * self.fov
            distance_to_wall = self.max_depth
            hit_color = (0, 150, 0)
            for depth in range(1, int(self.max_depth), 4):
                rx = self.x + depth * math.cos(ray_angle)
                ry = self.y + depth * math.sin(ray_angle)
                if rx < 0 or rx > self.w or ry < 0 or ry > self.h:
                    distance_to_wall = depth
                    hit_color = (0, 50, 0)
                    break
                if self._is_inside_obstacle(rx, ry):
                    distance_to_wall = depth
                    break
                treat_hit = False
                for t in self.treats:
                    if math.hypot(rx - t["x"], ry - t["y"]) < 20:
                        distance_to_wall = depth
                        hit_color = (255, 200, 0)
                        treat_hit = True
                        break
                if treat_hit:
                    break
                if math.hypot(rx - self.charger["x"], ry - self.charger["y"]) < 5:
                    distance_to_wall = depth
                    hit_color = (255, 0, 255)
                    break
                social_hit = False
                for s in self.social_targets:
                    if math.hypot(rx - s["x"], ry - s["y"]) < s["radius"]:
                        distance_to_wall = depth
                        hit_color = (80, 220, 255)
                        social_hit = True
                        break
                if social_hit:
                    break

            distance_to_wall *= math.cos(rad_heading - ray_angle)
            wall_height = int(min(128, 2000 / max(distance_to_wall, 1e-6)))
            shade = max(0.2, 1.0 - (distance_to_wall / self.max_depth))
            final_color = (int(hit_color[0] * shade), int(hit_color[1] * shade), int(hit_color[2] * shade))
            y1 = int(64 - wall_height / 2)
            y2 = int(64 + wall_height / 2)
            cv2.line(frame, (i, max(0, y1)), (i, min(127, y2)), final_color, 1)

        if self.laser_dist < self.max_depth:
            laser_h = int(min(128, 2000 / max(self.laser_dist, 1e-6)))
            ly = int(64 + laser_h / 2)
            cv2.circle(frame, (64, ly - 5), 3, (0, 0, 255), -1)

        noise = np.random.normal(0, 4, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        yy, xx = np.mgrid[0:128, 0:128]
        dist = np.sqrt((xx - 64) ** 2 + (yy - 64) ** 2) / 90.0
        vignette = np.clip(1.0 - dist * 0.25, 0.72, 1.0)
        frame = np.clip(frame.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)
        return frame

    def render_vision(self):
        frame = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for i in range(0, self.w, 64):
            cv2.line(frame, (i, 0), (i, self.h), (0, 40, 0), 1)
        for i in range(0, self.h, 64):
            cv2.line(frame, (0, i), (self.w, i), (0, 40, 0), 1)

        pulse = int(5 * math.sin(time.time() * 4))
        cv2.circle(frame, (int(self.charger["x"]), int(self.charger["y"])), self.charger["r"], (100, 0, 100), -1)
        cv2.circle(frame, (int(self.charger["x"]), int(self.charger["y"])), self.charger["r"] + pulse, (255, 0, 255), 2)

        for obs in self.obstacles:
            cv2.rectangle(frame, (obs["x"], obs["y"]), (obs["x"] + obs["w"], obs["y"] + obs["h"]), (0, 150, 0), 2)

        for t in self.treats:
            cv2.circle(frame, (int(t["x"]), int(t["y"])), 20, (255, 200, 0), -1)
            cv2.circle(frame, (int(t["x"]), int(t["y"])), 25, (255, 255, 0), 2)

        for s in self.social_targets:
            sx, sy = int(s["x"]), int(s["y"])
            cv2.circle(frame, (sx, sy), 20, (80, 220, 255), -1)
            cv2.circle(frame, (sx, sy - 18), 10, (200, 255, 255), -1)

        rad = math.radians(self.theta)
        pt1 = (int(self.x + 30 * math.cos(rad)), int(self.y + 30 * math.sin(rad)))
        pt2 = (int(self.x + 20 * math.cos(rad + 2.5)), int(self.y + 20 * math.sin(rad + 2.5)))
        pt3 = (int(self.x + 20 * math.cos(rad - 2.5)), int(self.y + 20 * math.sin(rad - 2.5)))
        cv2.fillPoly(frame, [np.array([pt1, pt2, pt3])], (0, 255, 255))

        laser_end = (int(self.x + 30 * math.cos(rad) + self.laser_dist * math.cos(rad)),
                     int(self.y + 30 * math.sin(rad) + self.laser_dist * math.sin(rad)))
        cv2.line(frame, pt1, laser_end, (0, 0, 255), 2)
        return frame


# ==========================================
# DRIVE SYSTEM
# ==========================================

class DriveSystem:
    def __init__(self):
        self.energy = 1.0
        self.stress = 0.08
        self.curiosity = 0.45
        self.social_bond = 0.12
        self.comfort = 0.68
        self.seek_drive = 0.35
        self.fatigue = 0.0
        self.pain_memory = 0.0
        self.familiarity = 0.0

    def as_vector(self):
        return np.array([
            self.energy,
            self.stress,
            self.curiosity,
            self.social_bond,
            self.comfort,
            self.seek_drive,
            self.fatigue,
            self.pain_memory
        ], dtype=np.float32)

    def step(self, dt, semantic_vision, audio_volume, touch, movement_mag, on_reward=False, on_pain=False, on_charger=False):
        face_present = float(semantic_vision[0])
        face_lock = float(semantic_vision[7])
        motion = float(semantic_vision[5])
        scene_change = float(semantic_vision[6])

        contact_load = float(max(touch[0], touch[1], touch[3])) if len(touch) >= 4 else 0.0
        front_closeness = (1.0 - float(touch[2])) if len(touch) >= 3 else 0.0
        touch_load = max(contact_load, front_closeness * 0.55)
        social_event = face_present * (0.5 + 0.5 * min(1.0, audio_volume * 4.0))

        self.energy = np.clip(self.energy - (0.0022 + movement_mag * 0.0015) * dt, 0.0, 1.0)
        self.fatigue = np.clip(self.fatigue + movement_mag * 0.0025 * dt - 0.008 * dt, 0.0, 1.0)

        boredom = 1.0 - min(1.0, (motion * 2.0 + audio_volume * 1.5 + touch_load * 2.0))
        self.curiosity = np.clip(self.curiosity + (boredom * 0.015 - 0.006 * face_present) * dt, 0.0, 1.0)

        self.familiarity = np.clip(self.familiarity + social_event * 0.01 * dt - 0.0005 * dt, 0.0, 1.0)
        self.social_bond = np.clip(
            self.social_bond + social_event * 0.02 * dt + face_lock * 0.01 * dt - 0.001 * dt,
            0.0, 1.0
        )

        stress_up = contact_load * 0.18 + front_closeness * 0.06 + max(0.0, audio_volume - 0.65) * 0.25 + max(0.0, scene_change - 0.22) * 0.08
        stress_down = 0.08 + 0.03 * self.comfort + 0.03 * self.social_bond + 0.02 * face_lock
        self.stress = np.clip(self.stress + (stress_up - stress_down) * dt, 0.0, 1.0)

        self.pain_memory = np.clip(self.pain_memory + (0.12 if on_pain else -0.025 * dt), 0.0, 1.0)

        self.comfort = np.clip(
            0.50 * (1.0 - self.stress)
            + 0.20 * (1.0 - min(1.0, scene_change * 2.5))
            + 0.15 * self.social_bond
            + 0.15 * face_lock,
            0.0, 1.0
        )

        self.seek_drive = np.clip(
            0.45 * self.curiosity + 0.25 * (1.0 - self.fatigue) + 0.20 * (1.0 - self.stress) + 0.10 * boredom,
            0.0, 1.0
        )

        if on_charger:
            self.energy = min(1.0, self.energy + 0.030 * dt)
            self.fatigue = max(0.0, self.fatigue - 0.040 * dt)
            self.stress = max(0.0, self.stress - 0.020 * dt)
            self.comfort = min(1.0, self.comfort + 0.020 * dt)

        if on_reward:
            self.energy = min(1.0, self.energy + 0.08)
            self.stress = max(0.0, self.stress - 0.10)
            self.comfort = min(1.0, self.comfort + 0.10)
            self.social_bond = min(1.0, self.social_bond + 0.02 * face_present)

        if on_pain:
            self.stress = min(1.0, self.stress + 0.12)
            self.seek_drive = max(0.0, self.seek_drive - 0.10)

    def mood_string(self):
        if self.energy < 0.18:
            return "DEPLETED"
        if self.stress > 0.78:
            return "ALARMED"
        if self.familiarity > 0.45 and self.social_bond > 0.35 and self.stress < 0.40:
            return "COMPANIONABLE"
        if self.social_bond > 0.55 and self.stress < 0.45:
            return "SOCIAL"
        if self.seek_drive > 0.62 and self.stress < 0.55:
            return "SEEKING"
        if self.comfort > 0.76:
            return "CALM"
        return "WATCHFUL"


# ==========================================
# VISION CORTEX
# ==========================================
class VisionCortex:
    def __init__(self, fovea_res=(192, 192), periphery_res=(256, 144), compact_fovea_res=(80, 80), compact_periphery_res=(96, 54)):
        self.fovea_res = fovea_res
        self.periphery_res = periphery_res
        self.compact_fovea_res = compact_fovea_res
        self.compact_periphery_res = compact_periphery_res
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
        self.upper_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")
        self.prev_gray_real = None
        self.prev_gray_sim = None

    def _extract_fovea(self, frame, pan_pos, tilt_pos):
        h, w = frame.shape[:2]
        cx = int((pan_pos + 1.0) * 0.5 * w)
        cy = int((tilt_pos + 1.0) * 0.5 * h)
        fw, fh = self.fovea_res
        half_w = fw // 2
        half_h = fh // 2
        x1 = max(0, min(w - fw, cx - half_w))
        y1 = max(0, min(h - fh, cy - half_h))
        crop = frame[y1:y1 + fh, x1:x1 + fw]
        if crop.shape[0] != fh or crop.shape[1] != fw:
            crop = cv2.resize(crop, self.fovea_res, interpolation=cv2.INTER_AREA)
        return crop

    def _detect_social_targets(self, gray):
        faces = []
        if not self.face_cascade.empty():
            frontal = self.face_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=6, minSize=(54, 54))
            if len(frontal) > 0:
                faces.extend(list(frontal))

        if not self.profile_cascade.empty():
            profile = self.profile_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=5, minSize=(54, 54))
            if len(profile) > 0:
                faces.extend(list(profile))

            flipped = cv2.flip(gray, 1)
            profile_r = self.profile_cascade.detectMultiScale(flipped, scaleFactor=1.10, minNeighbors=5, minSize=(54, 54))
            if len(profile_r) > 0:
                w = gray.shape[1]
                for (x, y, fw, fh) in profile_r:
                    faces.append((w - x - fw, y, fw, fh))

        uppers = []
        if not self.upper_cascade.empty():
            uppers = self.upper_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(88, 88))
            
        return faces, list(uppers)

    def _build_optic_bundle(self, fovea_bgr, periphery_bgr, motion_src_gray):
        fovea_small = cv2.resize(fovea_bgr, self.compact_fovea_res, interpolation=cv2.INTER_AREA)
        periph_small = cv2.resize(periphery_bgr, self.compact_periphery_res, interpolation=cv2.INTER_AREA)
        motion_small = cv2.resize(motion_src_gray, self.compact_periphery_res, interpolation=cv2.INTER_AREA)

        def opponent_pack(img_bgr):
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            r = img[..., 0]
            g = img[..., 1]
            b = img[..., 2]
            lum = np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0.0, 1.0)
            rg = np.clip((r - g) * 0.5 + 0.5, 0.0, 1.0)
            by = np.clip((b - 0.5 * (r + g)) * 0.5 + 0.5, 0.0, 1.0)
            return lum, rg, by

        f_lum, f_rg, f_by = opponent_pack(fovea_small)
        p_lum, p_rg, p_by = opponent_pack(periph_small)
        motion_norm = np.clip(motion_small.astype(np.float32) / 255.0, 0.0, 1.0)

        return np.concatenate([
            f_lum.flatten(), f_rg.flatten(), f_by.flatten(),
            p_lum.flatten(), p_rg.flatten(), p_by.flatten(),
            motion_norm.flatten()
        ]).astype(np.float32)

    def _score_candidate(self, gray, diff, box, weight):
        x, y, fw, fh = box
        h, w = gray.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + fw)
        y2 = min(h, y + fh)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        roi = gray[y1:y2, x1:x2]
        motion_roi = diff[y1:y2, x1:x2] if diff is not None else roi
        aspect = fw / max(1.0, fh)
        area = (fw * fh) / float(w * h)
        brightness = float(np.mean(roi) / 255.0)
        contrast = float(np.std(roi) / 255.0)
        motion = float(np.mean(motion_roi) / 255.0)
        center_x = (x + fw * 0.5) / w
        center_y = (y + fh * 0.5) / h
        center_bias = 1.0 - min(1.0, abs(center_x - 0.5) * 1.4 + abs(center_y - 0.5) * 1.1)
        aspect_score = max(0.0, 1.0 - abs(aspect - 0.72) * 1.4)
        size_score = np.clip(area * 14.0, 0.0, 1.0)
        detail_score = np.clip(contrast * 4.0, 0.0, 1.0)
        motion_score = np.clip(motion * 5.0, 0.0, 1.0)
        brightness_score = 1.0 - min(1.0, abs(brightness - 0.52) * 1.7)
        score = (
            0.30 * weight +
            0.18 * aspect_score +
            0.16 * size_score +
            0.12 * detail_score +
            0.10 * motion_score +
            0.08 * center_bias +
            0.06 * brightness_score
        )
        return float(np.clip(score, 0.0, 1.0))

    def process_real_world(self, frame, pan_pos, tilt_pos):
        h, w = frame.shape[:2]
        fovea = self._extract_fovea(frame, pan_pos, tilt_pos)
        periphery = cv2.resize(frame, self.periphery_res, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = 0.0
        scene_change = 0.0
        diff = np.zeros_like(gray)
        if self.prev_gray_real is not None and self.prev_gray_real.shape != gray.shape:
            self.prev_gray_real = None
        if self.prev_gray_real is not None:
            diff = cv2.absdiff(gray, self.prev_gray_real)
            motion = float(np.mean(diff) / 255.0)
            scene_change = float(np.std(diff) / 255.0)
        self.prev_gray_real = gray

        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges) / 255.0)
        brightness = float(np.mean(gray) / 255.0)
        contrast = float(np.std(gray) / 255.0)

        faces, uppers = self._detect_social_targets(gray)
        candidates = []
        for box in faces:
            candidates.append((box, 1.0))
        for box in uppers:
            candidates.append((box, 0.55))

        social_salience = 0.0
        face_x = 0.0
        face_y = 0.0
        face_size = 0.0
        face_lock = 0.0
        best_box = None
        best_score = 0.0

        if candidates:
            scored = []
            for box, weight in candidates:
                score = self._score_candidate(gray, diff, box, weight)
                scored.append((score, box, weight))
            best_score, best_box, best_weight = max(scored, key=lambda item: item[0])
            if best_score > 0.34:
                x, y, fw, fh = best_box
                cx = (x + fw * 0.5) / w
                cy = (y + fh * 0.5) / h
                social_salience = float(best_score)
                face_x = float((cx - 0.5) * 2.0)
                face_y = float((cy - 0.5) * 2.0)
                face_size = float((fw * fh) / float(w * h))
                tx = float(np.clip(pan_pos, -1.0, 1.0))
                ty = float(np.clip(tilt_pos, -1.0, 1.0))
                err = abs(face_x - tx) + abs(face_y - ty)
                face_lock = float(np.clip(1.0 - err * 0.45, 0.0, 1.0)) * social_salience
            else:
                best_box = None

        semantic = np.array([
            social_salience,
            face_x,
            face_y,
            face_size,
            brightness,
            motion,
            scene_change,
            face_lock,
            contrast,
            edge_density,
        ], dtype=np.float32)

        optic = self._build_optic_bundle(fovea, periphery, diff)
        valid_faces = [best_box] if best_box is not None else []
        return optic, semantic, valid_faces, [], fovea, periphery, diff

    def process_dream_state(self, ego_frame, pan_pos, tilt_pos, sim_social=None):
        fovea = cv2.resize(ego_frame, self.fovea_res, interpolation=cv2.INTER_AREA)
        periphery = cv2.resize(ego_frame, self.periphery_res, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(ego_frame, cv2.COLOR_BGR2GRAY)
        motion = 0.0
        scene_change = 0.0
        diff = np.zeros_like(gray)
        if self.prev_gray_sim is not None and self.prev_gray_sim.shape != gray.shape:
            self.prev_gray_sim = None
        if self.prev_gray_sim is not None:
            diff = cv2.absdiff(gray, self.prev_gray_sim)
            motion = float(np.mean(diff) / 255.0)
            scene_change = float(np.std(diff) / 255.0)
        self.prev_gray_sim = gray

        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges) / 255.0)
        brightness = float(np.mean(gray) / 255.0)
        contrast = float(np.std(gray) / 255.0)

        face_present = 0.0
        face_x = 0.0
        face_y = 0.0
        face_size = 0.0
        face_lock = 0.0
        if sim_social is not None:
            face_present = float(sim_social[0])
            face_x = float(sim_social[1])
            face_y = float(sim_social[2])
            face_size = float(sim_social[3])
            err = abs(face_x - float(np.clip(pan_pos, -1.0, 1.0))) + abs(face_y - float(np.clip(tilt_pos, -1.0, 1.0)))
            face_lock = float(np.clip(1.0 - err * 0.45, 0.0, 1.0)) * face_present

        semantic = np.array([
            face_present, face_x, face_y, face_size,
            brightness, motion, scene_change, face_lock,
            contrast, edge_density,
        ], dtype=np.float32)

        optic = self._build_optic_bundle(fovea, periphery, diff)
        return optic, semantic, fovea, periphery, diff


# ==========================================
# RAW CORTEX
# ==========================================

class RawCortex:
    def __init__(self):
        self.terminal_log = []
        self.is_alive = True
        self.shutdown_started = False
        self.embodiment = "REAL_WORLD"
        self.sim_world = SimWorld()

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_root = os.path.join(self.base_dir, "captain_v1_data")
        self.identity_dir = os.path.join(self.data_root, "identity")
        self.logs_dir = os.path.join(self.data_root, "logs")
        self.snapshots_dir = os.path.join(self.data_root, "snapshots")
        self.cache_dir = os.path.join(self.data_root, "cache")
        self.weights_dir = os.path.join(self.identity_dir, "weights")
        for d in (self.data_root, self.identity_dir, self.logs_dir, self.snapshots_dir, self.cache_dir, self.weights_dir):
            os.makedirs(d, exist_ok=True)

        self.db_path = os.path.join(self.identity_dir, "captain_v1.db")
        self.journal_path = os.path.join(self.logs_dir, "captain_journal_v1.log")
        self.core_spec_path = os.path.join(self.identity_dir, "captain_core_spec_v1.json")
        self.identity_profile_path = os.path.join(self.identity_dir, "captain_identity_profile_v1.json")
        self.weights_manifest_path = os.path.join(self.weights_dir, "weights_manifest_v1.json")
        
        self.session_start_ts = time.time()
        self.snapshot_interval = 1800.0
        self.weight_save_interval = 180.0
        self.last_weight_save_time = time.time()
        self.weight_save_thread = None
        self.shutdown_snapshot_enabled = False
        self.last_snapshot_time = time.time()
        self.flush_interval = 5.0
        self.last_flush_time = time.time()
        self.journal_buffer = []
        self.episode_buffer = []
        self.core_spec = dict(CORE_SPEC)

        self.compact_fovea_res = tuple(self.core_spec["compact_fovea_res"])
        self.compact_periphery_res = tuple(self.core_spec["compact_periphery_res"])
        self.motion_res = self.compact_periphery_res
        self.optic_size = (self.compact_fovea_res[0] * self.compact_fovea_res[1] * 3) + (self.compact_periphery_res[0] * self.compact_periphery_res[1] * 3) + (self.motion_res[0] * self.motion_res[1])
        self.audio_size = int(self.core_spec.get("audio_feature_bands", 192))
        self.audio_display_bands = int(self.core_spec.get("audio_display_bands", 64))
        self.audio_sample_rate = int(self.core_spec.get("audio_sample_rate", 22050))
        self.audio_fft_size = int(self.core_spec.get("audio_fft_size", 2048))
        self.audio_display = np.zeros(self.audio_display_bands, dtype=np.float32)
        self.audio_band_centers = np.linspace(0.0, self.audio_sample_rate / 2.0, self.audio_size, dtype=np.float32)
        self.audio_gain_reference = 1.0
        self.touch_size = 6
        self.semantic_size = 10
        self.drive_size = 8
        self.vocal_input_size = self.audio_size + self.semantic_size + self.drive_size

        self.visual_neurons = int(self.core_spec["visual_neurons"])
        self.audio_neurons = int(self.core_spec["audio_neurons"])
        self.touch_neurons = int(self.core_spec["touch_neurons"])
        self.associative_neurons = int(self.core_spec["associative_neurons"])
        self.vocal_neurons = int(self.core_spec["vocal_neurons"])
        self.total_hidden = self.visual_neurons + self.audio_neurons + self.touch_neurons + self.vocal_neurons + self.associative_neurons
        self.num_outputs = 6

        self.total_synapses = (
            self.optic_size * self.visual_neurons +
            self.audio_size * self.audio_neurons +
            self.touch_size * self.touch_neurons +
            self.vocal_input_size * self.vocal_neurons +
            (self.semantic_size + self.drive_size) * self.associative_neurons +
            self.total_hidden * self.associative_neurons +
            self.associative_neurons * self.num_outputs
        )

        self.visual_hidden = np.zeros(self.visual_neurons, dtype=np.float32)
        self.audio_hidden = np.zeros(self.audio_neurons, dtype=np.float32)
        self.touch_hidden = np.zeros(self.touch_neurons, dtype=np.float32)
        self.vocal_hidden = np.zeros(self.vocal_neurons, dtype=np.float32)
        self.associative_hidden = np.zeros(self.associative_neurons, dtype=np.float32)
        self.output_layer = np.zeros(self.num_outputs, dtype=np.float32)
        self.desired_output = np.zeros(self.num_outputs, dtype=np.float32)

        self.raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.ego_frame = np.zeros((128, 128, 3), dtype=np.uint8)
        self.fovea_frame = np.zeros((128, 128, 3), dtype=np.uint8)
        self.periphery_frame = np.zeros((90, 160, 3), dtype=np.uint8)

        self.optic_nerve = np.zeros(self.optic_size, dtype=np.float32)
        self.audio_nerve = np.zeros(self.audio_size, dtype=np.float32)
        self.touch_nerve = np.zeros(self.touch_size, dtype=np.float32)
        self.semantic_vision = np.zeros(self.semantic_size, dtype=np.float32)
        self.audio_volume = 0.0

        self.pan_pos = 0.0
        self.tilt_pos = 0.0
        self.detected_faces = []

        self.drives = DriveSystem()
        self.internal_state = np.array([1.0, 0.08, 0.45], dtype=np.float32)
        self.camera_status = "INITIALIZING..."
        self.reward_flash = 0.0

        self.virtual_body_x = 512.0
        self.virtual_body_y = 512.0
        self.virtual_body_theta = 0.0

        self.vision = VisionCortex(
            fovea_res=(192, 192),
            periphery_res=(256, 144),
            compact_fovea_res=self.compact_fovea_res,
            compact_periphery_res=self.compact_periphery_res
        )

        self.db_lock = threading.Lock()
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()
        self._write_core_spec_file()

        self.vocal_queue = []
        self.heard_prototypes = []
        self.last_vocal_time = 0.0
        self.is_booting = True
        self.boot_timer = time.time()
        self.boot_step = 0

        self.behavior_state = "resting"
        self.behavior_timer = 0.0
        self.last_face_seen_time = 0.0
        self.face_presence_smooth = 0.0
        self.face_lock_smooth = 0.0
        self.motion_smooth = 0.0
        self.audio_smooth = 0.0
        self.voice_smooth = 0.0
        self.presence_hold = 0.0
        self.absence_hold = 0.0
        self.greet_cooldown = 0.0
        self.companion_timer = 0.0
        self.idle_scan_phase = 0.0
        self.last_body_x = self.virtual_body_x
        self.last_body_y = self.virtual_body_y
        self.stuck_timer = 0.0
        self.escape_turn_sign = 1.0
        self.roam_bias = 0.0
        self.roam_bias_timer = 0.0
        self.social_candidate_hold = 0.0
        self.familiarity_score = 0.0
        self.gaze_mode = "center"
        self.gaze_target_x = 0.0
        self.gaze_target_y = 0.0
        self.saccade_timer = 0.0
        self.fixation_timer = 0.0
        self.microsaccade_phase = 0.0
        self.loop_score = 0.0
        self.last_turn_sign = 0.0
        self.visit_history = []
        self.roam_heading = random.uniform(0.0, 360.0)
        self.roam_goal_timer = 0.0
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0
        self.body_speed = 0.0
        self.body_turn_rate = 0.0
        self.sim_treat_signal = 0.0
        self.sim_treat_angle = 0.0
        self.sim_charge_signal = 0.0
        self.sim_charge_angle = 0.0
        self.sim_open_space = 1.0

        self.eye_sources = ["WEBCAM", "DESKTOP", "PANEL", "DREAM_STATE", "FUTURE_BODY"]
        self.active_eye_source = "WEBCAM"
        self.last_real_eye_source = "WEBCAM"
        self.desktop_capture_interval = 0.12
        self.panel_render_interval = 0.10
        self.last_desktop_capture_time = 0.0
        self.last_panel_render_time = 0.0
        self.desktop_eye_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        self.panel_eye_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        self.future_body_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        self.manual_eye_override = "WEBCAM"
        self.autonomous_attention_enabled = False
        self.autonomous_eye_target = "WEBCAM"
        self.panel_hover_target = None
        self.panel_hover_time = 0.0
        self.panel_commit_cooldown = 0.0
        self.panel_dwell_seconds = 0.72
        self.last_panel_action_time = 0.0
        self.panel_action_log = []
        
        # UI Themes Setup
        self.themes = [
            {"name": "Cyan", "accent": (120, 255, 220), "accent2": (0, 255, 255)},
            {"name": "Purple", "accent": (210, 120, 255), "accent2": (160, 110, 255)},
            {"name": "Amber", "accent": (255, 190, 90), "accent2": (255, 150, 60)},
            {"name": "Lime", "accent": (150, 255, 120), "accent2": (90, 220, 90)},
            {"name": "Rose", "accent": (255, 120, 165), "accent2": (255, 80, 130)},
            {"name": "Ice", "accent": (180, 235, 255), "accent2": (120, 210, 255)},
            {"name": "Sunset", "accent": (255, 145, 110), "accent2": (255, 95, 70)},
            {"name": "Retro", "accent": (255, 235, 120), "accent2": (255, 180, 40)},
            {"name": "Toxic", "accent": (170, 255, 90), "accent2": (110, 255, 40)},
            {"name": "Ghost", "accent": (210, 210, 230), "accent2": (140, 170, 255)},
        ]
        self.theme_index = 0
        self.avatar_theme = self.themes[self.theme_index]["name"]
        
        self.desktop_mouse_norm = np.array([0.0, 0.0], dtype=np.float32)
        
        self.control_line_height = 76
        self.control_line_frame = np.zeros((self.control_line_height, 640, 3), dtype=np.uint8)
        self.control_line_hover_target = None
        self.control_line_hover_time = 0.0
        self.control_line_dwell_seconds = 0.64
        self._render_panel_eye()
        self._render_control_line()

    def _current_theme(self):
        return self.themes[self.theme_index % len(self.themes)]

    def _cycle_theme(self, delta, reason="manual"):
        self.theme_index = (self.theme_index + delta) % len(self.themes)
        self.avatar_theme = self._current_theme()["name"]
        self.log(f"[SHELL] Theme -> {self.avatar_theme} ({reason})")
        self._render_panel_eye()
        self._render_control_line()

    def _current_focus_vector(self):
        if self.active_eye_source == "DESKTOP":
            fx = float(np.clip(self.desktop_mouse_norm[0], -1.0, 1.0))
            fy = float(np.clip(self.desktop_mouse_norm[1], -1.0, 1.0))
            return fx * 0.85, fy * 0.75
        if self.face_presence_smooth > 0.20:
            fx = float(np.clip(self.semantic_vision[1], -1.0, 1.0))
            fy = float(np.clip(self.semantic_vision[2], -1.0, 1.0))
            return fx * 0.70, fy * 0.60
        return float(np.clip(self.pan_pos, -1.0, 1.0)) * 0.55, float(np.clip(self.tilt_pos, -1.0, 1.0)) * 0.45

    def _update_desktop_mouse_norm(self):
        try:
            app = QApplication.instance()
            if app is None:
                return
            screen = app.primaryScreen()
            if screen is None:
                return
            rect = screen.availableGeometry()
            pos = QCursor.pos()
            if rect.width() <= 1 or rect.height() <= 1:
                return
            nx = ((pos.x() - rect.x()) / rect.width()) * 2.0 - 1.0
            ny = ((pos.y() - rect.y()) / rect.height()) * 2.0 - 1.0
            self.desktop_mouse_norm = np.array([np.clip(nx, -1.0, 1.0), np.clip(ny, -1.0, 1.0)], dtype=np.float32)
        except Exception:
            pass

    def _apply_gaze_crosshair_to_frame(self, frame):
        if frame is None or frame.size == 0:
            return frame
        out = frame.copy()
        h, w = out.shape[:2]
        if self.active_eye_source == "DESKTOP":
            cx = int((float(self.desktop_mouse_norm[0]) + 1.0) * 0.5 * w)
            cy = int((float(self.desktop_mouse_norm[1]) + 1.0) * 0.5 * h)
        else:
            cx = int((float(self.pan_pos) + 1.0) * 0.5 * w)
            cy = int((float(self.tilt_pos) + 1.0) * 0.5 * h)
        cx = max(10, min(w - 10, cx))
        cy = max(10, min(h - 10, cy))
        color = (70, 255, 120)
        if self.active_eye_source == "DESKTOP":
            cv2.circle(out, (cx, cy), 7, color, 1)
            cv2.line(out, (cx - 10, cy), (cx + 10, cy), color, 1)
            cv2.line(out, (cx, cy - 10), (cx, cy + 10), color, 1)
        else:
            cv2.line(out, (cx - 7, cy - 7), (cx + 7, cy + 7), color, 2)
            cv2.line(out, (cx - 7, cy + 7), (cx + 7, cy - 7), color, 2)
        return out

    def _panel_buttons(self):
        # Spaced out beautifully so nothing overlaps
        return [
            {"label": "WEBCAM", "source": "WEBCAM", "rect": (40, 218, 170, 54)},
            {"label": "DESKTOP", "source": "DESKTOP", "rect": (240, 218, 170, 54)},
            {"label": "PANEL", "source": "PANEL", "rect": (440, 218, 150, 54)},
            {"label": "DREAM", "source": "DREAM_STATE", "rect": (140, 288, 170, 44)},
            {"label": "FUTURE", "source": "FUTURE_BODY", "rect": (360, 288, 170, 44)},
            {"label": "<", "source": "THEME_PREV", "rect": (408, 156, 30, 28)},
            {"label": ">", "source": "THEME_NEXT", "rect": (562, 156, 30, 28)},
        ]

    def _control_line_buttons(self):
        return [
            {"label": "W", "source": "WEBCAM", "rect": (170, 18, 62, 36)},
            {"label": "D", "source": "DESKTOP", "rect": (244, 18, 62, 36)},
            {"label": "P", "source": "PANEL", "rect": (318, 18, 62, 36)},
            {"label": "R", "source": "DREAM_STATE", "rect": (392, 18, 62, 36)},
            {"label": "F", "source": "FUTURE_BODY", "rect": (466, 18, 62, 36)},
        ]

    def _apply_eye_source(self, source, reason="manual"):
        if source not in getattr(self, "eye_sources", []):
            return
        if source == "DREAM_STATE":
            self.embodiment = "DREAM_STATE"
        else:
            self.embodiment = "REAL_WORLD"
            self.last_real_eye_source = source
        changed = (source != getattr(self, "active_eye_source", None))
        self.active_eye_source = source
        if changed:
            suffix = f" ({reason})" if reason else ""
            self.log(f"[VISION] Eye source -> {source}{suffix}")

    def _set_active_eye_source(self, source, origin="manual"):
        if source not in getattr(self, "eye_sources", []):
            return
        if origin == "manual":
            self.manual_eye_override = source
            self.autonomous_attention_enabled = False
        else:
            self.autonomous_eye_target = source
        self._apply_eye_source(source, reason=origin)

    def _toggle_attention_mode(self):
        self.autonomous_attention_enabled = not self.autonomous_attention_enabled
        if self.autonomous_attention_enabled:
            self.manual_eye_override = None
            self.log("[VISION] Autonomous attention -> ON")
            self._apply_eye_source(self.autonomous_eye_target, reason="auto")
        else:
            self.manual_eye_override = self.active_eye_source
            self.log(f"[VISION] Autonomous attention -> OFF ({self.manual_eye_override})")

    def _capture_desktop_eye(self):
        try:
            self._update_desktop_mouse_norm()
            app = QApplication.instance()
            if app is None:
                return self.desktop_eye_frame
            screen = app.primaryScreen()
            if screen is None:
                return self.desktop_eye_frame
            pix = screen.grabWindow(0)
            if pix.isNull():
                return self.desktop_eye_frame
            img = pix.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
            w, h = img.width(), img.height()
            ptr = img.bits()
            size = img.sizeInBytes() if hasattr(img, 'sizeInBytes') else img.byteCount()
            if hasattr(ptr, 'asstring'):
                buf = ptr.asstring(size)
            else:
                try:
                    ptr.setsize(size)
                except Exception:
                    pass
                buf = bytes(ptr)
            arr = np.frombuffer(buf, np.uint8).reshape((h, w, 4))
            frame = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            if frame is not None and frame.size > 0 and int(np.mean(frame)) > 0:
                self.desktop_eye_frame = frame.copy()
            return self.desktop_eye_frame
        except Exception as exc:
            pass
        return self.desktop_eye_frame

    def _render_control_line(self):
        h = self.control_line_height
        frame = np.zeros((h, 640, 3), dtype=np.uint8)
        frame[:] = (8, 12, 22)
        cv2.rectangle(frame, (0, 0), (639, h-1), (18, 34, 58), 2)
        mood = self.drives.mood_string()
        world_eye = self.active_eye_source
        theme_name = self.avatar_theme.upper()
        cv2.putText(frame, "CTRL", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
        cv2.putText(frame, f"EYE:{world_eye}", (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120, 255, 180), 1)
        cv2.putText(frame, f"MOOD:{mood}", (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 255, 180), 1)
        for btn in self._control_line_buttons():
            x, y, w, bh = btn["rect"]
            source = btn["source"]
            active = (source == self.active_eye_source)
            hovered = (source == self.control_line_hover_target)
            color = (0, 220, 220) if active else (60, 100, 140)
            if hovered:
                color = (80, 255, 120)
            cv2.rectangle(frame, (x, y), (x+w, y+bh), color, 2)
            if hovered and self.control_line_hover_time > 0.0:
                fill_w = int(w * min(1.0, self.control_line_hover_time / max(0.01, self.control_line_dwell_seconds)))
                cv2.rectangle(frame, (x+2, y+bh-7), (x+2+fill_w, y+bh-3), (80,255,120), -1)
            cv2.putText(frame, btn["label"], (x+20, y+24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
        
        cv2.putText(frame, f"THEME:{theme_name}", (520, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self._current_theme()["accent"], 1)
        self.control_line_frame = frame
        return frame

    def _apply_control_line_to_frame(self, frame):
        if frame is None or frame.size == 0:
            return frame
        overlay = self._render_control_line()
        out = frame.copy()
        oh = min(self.control_line_height, out.shape[0])
        ow = out.shape[1]
        resized = cv2.resize(overlay, (ow, oh), interpolation=cv2.INTER_AREA)
        out[out.shape[0]-oh:out.shape[0], 0:ow] = resized
        return out

    def process_manual_click(self, x, y):
        # We manually clicked the frame (via UI thread)
        # Check control line first
        if self.active_eye_source != "PANEL" and y > 480 - self.control_line_height:
            cy = y - (480 - self.control_line_height)
            for btn in self._control_line_buttons():
                bx, by, bw, bh = btn["rect"]
                if bx <= x <= bx + bw and by <= cy <= by + bh:
                    self._commit_button_action(btn["source"], "manual_click")
                    return
        
        # Check panel buttons
        if self.active_eye_source == "PANEL":
            for btn in self._panel_buttons():
                bx, by, bw, bh = btn["rect"]
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    self._commit_button_action(btn["source"], "manual_click")
                    return

    def _commit_button_action(self, source, method="control"):
        self.panel_commit_cooldown = 1.1
        self.last_panel_action_time = time.time()
        if source == "THEME_PREV":
            self._cycle_theme(-1, reason=method)
            self.inject_reward() # Micro reward for learning GUI
        elif source == "THEME_NEXT":
            self._cycle_theme(1, reason=method)
            self.inject_reward()
        else:
            self.panel_action_log.append((self.last_panel_action_time, source))
            if len(self.panel_action_log) > 8:
                self.panel_action_log = self.panel_action_log[-8:]
            if self.autonomous_attention_enabled and self.manual_eye_override is None:
                self.autonomous_eye_target = source
                self._apply_eye_source(source, reason=method)
            else:
                self.manual_eye_override = source
                self._apply_eye_source(source, reason=method)
            self.log(f"[UI] Commit -> {source}")
            self.inject_reward() # Tiny dopamine hit trains him to click things!

    def _update_autonomous_attention(self, dt):
        self.panel_commit_cooldown = max(0.0, self.panel_commit_cooldown - dt)
        target = self.autonomous_eye_target
        if self.face_presence_smooth > 0.30 or self.face_lock_smooth > 0.22:
            target = "WEBCAM"
        elif self.behavior_state in ("foraging", "seeking_charge") or self.embodiment == "DREAM_STATE":
            target = "DREAM_STATE"
        elif self.behavior_state in ("investigating", "settling") and self.drives.curiosity > 0.35:
            target = "DESKTOP"
        elif self.behavior_state in ("roaming", "resting") and self.drives.curiosity > 0.42 and self.face_presence_smooth < 0.18:
            target = "DREAM_STATE"
        else:
            target = "WEBCAM"
        self.autonomous_eye_target = target
        if self.autonomous_attention_enabled and self.manual_eye_override is None and self.active_eye_source != "PANEL":
            self._apply_eye_source(target, reason="auto")

    def _update_panel_dwell_select(self, dt):
        frame = None
        buttons = None
        hovered_attr = "control_line_hover_target"
        hover_time_attr = "control_line_hover_time"
        dwell_seconds = self.control_line_dwell_seconds

        if self.active_eye_source == "PANEL":
            frame = self.panel_eye_frame
            buttons = self._panel_buttons()
            hovered_attr = "panel_hover_target"
            hover_time_attr = "panel_hover_time"
            dwell_seconds = self.panel_dwell_seconds
        else:
            frame = self.control_line_frame
            buttons = self._control_line_buttons()

        frame_h, frame_w = frame.shape[:2]
        cx = int((self.pan_pos + 1.0) * 0.5 * frame_w)
        cy = int((self.tilt_pos + 1.0) * 0.5 * frame_h)
        cx = max(0, min(frame_w - 1, cx))
        cy = max(0, min(frame_h - 1, cy))
        hovered = None
        for btn in buttons:
            x, y, w, h = btn["rect"]
            if x <= cx <= x + w and y <= cy <= y + h:
                hovered = btn["source"]
                break

        current_hover = getattr(self, hovered_attr)
        current_time = getattr(self, hover_time_attr)
        if hovered != current_hover:
            setattr(self, hovered_attr, hovered)
            setattr(self, hover_time_attr, 0.0)
            if hovered is not None:
                tag = "PANEL" if self.active_eye_source == "PANEL" else "CONTROL"
            return

        if hovered is None:
            setattr(self, hover_time_attr, 0.0)
            return

        current_time += dt
        setattr(self, hover_time_attr, current_time)
        if current_time >= dwell_seconds and self.panel_commit_cooldown <= 0.0:
            setattr(self, hover_time_attr, 0.0)
            self._commit_button_action(hovered, "dwell")

    def _render_panel_eye(self):
        theme = self._current_theme()
        accent = theme["accent"]
        accent2 = theme["accent2"]
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:] = (10, 14, 24)
        cv2.rectangle(frame, (20, 20), (620, 340), (20, 35, 55), 2)
        cv2.putText(frame, "CAPTAIN CONTROL ROOM", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        mood = self.drives.mood_string()
        mode_txt = "AUTO" if self.autonomous_attention_enabled and self.manual_eye_override is None else "MANUAL"
        cv2.putText(frame, f"MODE: {mode_txt}", (30, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 180), 2)
        cv2.putText(frame, f"AUTO TARGET: {self.autonomous_eye_target}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 255, 180), 1)
        cv2.putText(frame, "CONTROL LINE: ALWAYS VISIBLE", (30, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        cv2.putText(frame, f"MOOD: {mood}", (30, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (120, 255, 180), 2)
        cv2.putText(frame, f"BEHAVIOR: {self.behavior_state.upper()}", (30, 166), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (120, 255, 180), 2)
        cv2.putText(frame, "THEME", (360, 147), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1)
        cv2.rectangle(frame, (440, 152), (560, 184), (40, 80, 110), 1)
        cv2.putText(frame, theme["name"].upper(), (455, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.48, accent, 1)

        buttons = self._panel_buttons()
        cx = int((self.pan_pos + 1.0) * 0.5 * frame.shape[1])
        cy = int((self.tilt_pos + 1.0) * 0.5 * frame.shape[0])
        cx = max(10, min(frame.shape[1]-10, cx))
        cy = max(10, min(frame.shape[0]-10, cy))
        for btn in buttons:
            label = btn["label"]
            source = btn["source"]
            x, y, w, h = btn["rect"]
            active = (source == self.active_eye_source)
            hovered = (source == self.panel_hover_target)
            color = (0, 220, 220) if active else (60, 100, 140)
            if source in ("THEME_PREV", "THEME_NEXT"):
                color = accent2
            if hovered:
                color = (80, 255, 120)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            if hovered and self.panel_hover_time > 0.0:
                fill_w = int(w * min(1.0, self.panel_hover_time / max(0.01, self.panel_dwell_seconds)))
                cv2.rectangle(frame, (x+2, y+h-8), (x+2+fill_w, y+h-3), (80,255,120), -1)
            tx = x + (8 if source.startswith("THEME") else 12)
            scale = 0.70 if source.startswith("THEME") else 0.65
            yoff = y + 22 if source.startswith("THEME") else y + 34
            cv2.putText(frame, label, (tx, yoff), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
            
        if self.panel_action_log:
            recent = self.panel_action_log[-2:]
            ay = 46
            for ts, action in recent:
                cv2.putText(frame, f"{time.strftime('%H:%M:%S', time.localtime(ts))} {action}", (360, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 255, 180), 1)
                ay += 16
                
        # Draw interactive crosshair
        if self.panel_hover_target is not None:
            cv2.rectangle(frame, (cx-10, cy-10), (cx+10, cy+10), (80,255,120), 1)
            cv2.circle(frame, (cx, cy), 2, (80,255,120), -1)
        else:
            cv2.line(frame, (cx-8, cy), (cx+8, cy), (60,255,120), 2)
            cv2.line(frame, (cx, cy-8), (cx, cy+8), (60,255,120), 2)
        
        self.panel_eye_frame = frame
        return frame

    def start_hardware(self):
        self.log(f"[INIT] Waking up Hardware Sensors... DB: {self.db_path}")
        self.log("[SYS] Entering V32-MAX branch. Expanded sensory + cortex capacity online.")
        threading.Thread(target=self._run_optics, daemon=True).start()
        threading.Thread(target=self._run_audio_in, daemon=True).start()
        threading.Thread(target=self._run_audio_out, daemon=True).start()

    def _write_core_spec_file(self):
        try:
            profile = {
                "app_version": APP_VERSION,
                "schema_version": SCHEMA_VERSION,
                "identity_version": IDENTITY_VERSION,
                "db_file": os.path.basename(self.db_path),
                "journal_file": os.path.basename(self.journal_path),
                "session_start": self.session_start_ts,
                "audio_feature_bands": self.audio_size,
            }
            with open(self.core_spec_path, "w", encoding="utf-8") as f:
                json.dump(self.core_spec, f, indent=2)
            with open(self.identity_profile_path, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2)
        except Exception:
            pass

    def _queue_journal(self, level, event, message, data=None):
        entry = {
            "ts": time.time(),
            "level": level,
            "event": event,
            "message": message,
            "data": data or {},
        }
        self.journal_buffer.append(entry)
        try:
            with open(self.journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _queue_episode(self, reward=0.0):
        self.episode_buffer.append((
            time.time(),
            self.embodiment,
            self.behavior_state,
            float(self.semantic_vision[0]),
            float(self.semantic_vision[7]),
            float(self.audio_volume),
            float(np.max(self.touch_nerve[:4])),
            float(reward),
            float(self.drives.stress),
            float(self.drives.energy),
        ))

    def _flush_buffers(self, force=False):
        now = time.time()
        if not force and (now - self.last_flush_time) < self.flush_interval:
            return
        with self.db_lock:
            c = self.db.cursor()
            try:
                if self.journal_buffer:
                    c.executemany(
                        "INSERT INTO journal (ts, level, event, message, data_json) VALUES (?, ?, ?, ?, ?)",
                        [(e["ts"], e["level"], e["event"], e["message"], json.dumps(e["data"], ensure_ascii=False)) for e in self.journal_buffer]
                    )
                    self.journal_buffer.clear()
                if self.episode_buffer:
                    c.executemany(
                        "INSERT INTO episodes (ts, embodiment, behavior, face_present, face_lock, audio_volume, touch_peak, reward, stress, energy) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        self.episode_buffer
                    )
                    self.episode_buffer.clear()
                self.db.commit()
                self.last_flush_time = now
            except Exception:
                self.db.rollback()

    def create_snapshot(self, label="manual"):
        ts = time.strftime("%Y%m%d_%H%M%S")
        snap_dir = os.path.join(self.snapshots_dir, f"{ts}_{label}")
        os.makedirs(snap_dir, exist_ok=True)
        db_copy = os.path.join(snap_dir, os.path.basename(self.db_path))
        try:
            self._flush_buffers(force=True)
            with self.db_lock:
                snap_db = sqlite3.connect(db_copy)
                with self.db:
                    self.db.backup(snap_db)
                snap_db.close()
            shutil.copy2(self.core_spec_path, os.path.join(snap_dir, os.path.basename(self.core_spec_path)))
            shutil.copy2(self.identity_profile_path, os.path.join(snap_dir, os.path.basename(self.identity_profile_path)))
            if os.path.exists(self.journal_path):
                shutil.copy2(self.journal_path, os.path.join(snap_dir, os.path.basename(self.journal_path)))
            manifest = {
                "app_version": APP_VERSION,
                "schema_version": SCHEMA_VERSION,
                "label": label,
                "created_ts": time.time(),
                "db_file": os.path.basename(db_copy),
                "journal_file": os.path.basename(self.journal_path),
            }
            with open(os.path.join(snap_dir, "snapshot_manifest_v1.json"), "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, indent=2)
            with self.db_lock:
                c = self.db.cursor()
                c.execute("INSERT INTO snapshots (ts, label, path, db_bytes) VALUES (?, ?, ?, ?)", (time.time(), label, db_copy, os.path.getsize(db_copy)))
                self.db.commit()
            self.log(f"[SNAPSHOT] Saved {label} snapshot.")
            self._queue_journal("INFO", "snapshot", f"Snapshot created: {label}", {"path": db_copy})
            self.last_snapshot_time = time.time()
        except Exception as exc:
            self.log(f"[SNAPSHOT] Failed: {exc}")


    def _weight_file_map(self):
        return {
            "W_visual": os.path.join(self.weights_dir, "W_visual_fp16.npy"),
            "W_audio": os.path.join(self.weights_dir, "W_audio_fp16.npy"),
            "W_touch": os.path.join(self.weights_dir, "W_touch_fp16.npy"),
            "W_vocal": os.path.join(self.weights_dir, "W_vocal_fp16.npy"),
            "W_state": os.path.join(self.weights_dir, "W_state_fp16.npy"),
            "W_assoc": os.path.join(self.weights_dir, "W_assoc_fp16.npy"),
            "W_motor": os.path.join(self.weights_dir, "W_motor_fp16.npy"),
        }

    def _write_weight_manifest(self):
        manifest = {
            "app_version": APP_VERSION,
            "schema_version": SCHEMA_VERSION,
            "core_spec": self.core_spec,
            "updated_ts": time.time(),
            "files": {k: os.path.basename(v) for k, v in self._weight_file_map().items()},
        }
        with open(self.weights_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def _init_random_weights(self):
        self.W_visual = (np.random.randn(self.optic_size, self.visual_neurons) * np.sqrt(1 / self.optic_size)).astype(np.float32)
        self.W_audio = (np.random.randn(self.audio_size, self.audio_neurons) * np.sqrt(1 / self.audio_size)).astype(np.float32)
        self.W_touch = (np.random.randn(self.touch_size, self.touch_neurons) * np.sqrt(1 / self.touch_size)).astype(np.float32)
        self.W_vocal = (np.random.randn(self.vocal_input_size, self.vocal_neurons) * np.sqrt(1 / self.vocal_input_size)).astype(np.float32)
        self.W_state = (np.random.randn(self.semantic_size + self.drive_size, self.associative_neurons) * np.sqrt(1 / (self.semantic_size + self.drive_size))).astype(np.float32)
        self.W_assoc = (np.random.randn(self.total_hidden, self.associative_neurons) * np.sqrt(1 / self.total_hidden)).astype(np.float32)
        self.W_motor = (np.random.randn(self.associative_neurons, self.num_outputs) * np.sqrt(1 / self.associative_neurons)).astype(np.float32)

    def _save_weight_bundle_sync(self):
        file_map = self._weight_file_map()
        tmp_manifest = None
        for name, fp in file_map.items():
            arr = getattr(self, name)
            np.save(fp, arr.astype(np.float16), allow_pickle=False)
        self._write_weight_manifest()

    def _load_weight_bundle_sync(self):
        file_map = self._weight_file_map()
        self.W_visual = np.load(file_map["W_visual"], allow_pickle=False).astype(np.float32).reshape((self.optic_size, self.visual_neurons))
        self.W_audio = np.load(file_map["W_audio"], allow_pickle=False).astype(np.float32).reshape((self.audio_size, self.audio_neurons))
        self.W_touch = np.load(file_map["W_touch"], allow_pickle=False).astype(np.float32).reshape((self.touch_size, self.touch_neurons))
        self.W_vocal = np.load(file_map["W_vocal"], allow_pickle=False).astype(np.float32).reshape((self.vocal_input_size, self.vocal_neurons))
        self.W_state = np.load(file_map["W_state"], allow_pickle=False).astype(np.float32).reshape((self.semantic_size + self.drive_size, self.associative_neurons))
        self.W_assoc = np.load(file_map["W_assoc"], allow_pickle=False).astype(np.float32).reshape((self.total_hidden, self.associative_neurons))
        self.W_motor = np.load(file_map["W_motor"], allow_pickle=False).astype(np.float32).reshape((self.associative_neurons, self.num_outputs))

    def _ensure_matrix_metadata_schema(self, c):
        cols = [row[1] for row in c.execute("PRAGMA table_info(matrix)").fetchall()]
        wanted = {"id", "bundle_dir", "manifest_path", "updated_ts"}
        if set(cols) == wanted:
            return
        if cols:
            c.execute("ALTER TABLE matrix RENAME TO matrix_legacy")
        c.execute("""
            CREATE TABLE IF NOT EXISTS matrix (
                id INTEGER PRIMARY KEY,
                bundle_dir TEXT,
                manifest_path TEXT,
                updated_ts REAL
            )
        """)

    def _init_db(self):
        with self.db_lock:
            c = self.db.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    embodiment TEXT,
                    behavior TEXT,
                    face_present REAL,
                    face_lock REAL,
                    audio_volume REAL,
                    touch_peak REAL,
                    reward REAL,
                    stress REAL,
                    energy REAL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    level TEXT,
                    event TEXT,
                    message TEXT,
                    data_json TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL,
                    label TEXT,
                    path TEXT,
                    db_bytes INTEGER
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS identity_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS familiarity (
                    entity_key TEXT PRIMARY KEY,
                    score REAL,
                    last_seen REAL,
                    notes TEXT
                )
            """)
            self._ensure_matrix_metadata_schema(c)
            self.db.commit()

            ident = c.execute("SELECT value FROM identity_meta WHERE key='captain_uuid'").fetchone()
            if ident is None:
                c.execute("INSERT OR REPLACE INTO identity_meta (key, value) VALUES (?, ?)", ("captain_uuid", str(uuid.uuid4())))
                c.execute("INSERT OR REPLACE INTO identity_meta (key, value) VALUES (?, ?)", ("developmental_stage", "prototype_v32_max"))
                c.execute("INSERT OR REPLACE INTO identity_meta (key, value) VALUES (?, ?)", ("schema_version", str(SCHEMA_VERSION)))
                c.execute("INSERT OR REPLACE INTO identity_meta (key, value) VALUES (?, ?)", ("app_version", APP_VERSION))
                c.execute("INSERT OR REPLACE INTO identity_meta (key, value) VALUES (?, ?)", ("core_spec", json.dumps(self.core_spec)))
                self.db.commit()

            needs_reset = True
            loaded_from_legacy = False
            try:
                meta = c.execute("SELECT bundle_dir, manifest_path, updated_ts FROM matrix WHERE id=1").fetchone()
            except sqlite3.OperationalError:
                meta = None
            bundle_exists = all(os.path.exists(fp) for fp in self._weight_file_map().values())
            if meta and bundle_exists:
                try:
                    self._load_weight_bundle_sync()
                    needs_reset = False
                    self.log(f"[MEMORY] External weight bundle loaded securely.")
                except Exception:
                    self.log("[SYS] Weight bundle mismatch. Generating new cortex.")
                    needs_reset = True
            elif bundle_exists:
                try:
                    self._load_weight_bundle_sync()
                    needs_reset = False
                    self.log(f"[MEMORY] Weight files found and loaded.")
                except Exception:
                    needs_reset = True

        if needs_reset:
            self._init_random_weights()
            self._save_weight_bundle_sync()
            self.log(f"[MEMORY] Newborn entity. {self.total_synapses:,} synapses initialized.")
        elif loaded_from_legacy:
            self._save_weight_bundle_sync()

        with self.db_lock:
            c = self.db.cursor()
            c.execute(
                "INSERT OR REPLACE INTO matrix (id, bundle_dir, manifest_path, updated_ts) VALUES (1, ?, ?, ?)",
                (self.weights_dir, self.weights_manifest_path, time.time())
            )
            self.db.commit()
        self._queue_journal("INFO", "boot", "Identity DB initialized", {"db": self.db_path, "total_synapses": self.total_synapses})

    def _background_save(self):
        with self.save_lock:
            try:
                save_ts = time.time()
                self._save_weight_bundle_sync()
                db_bg = sqlite3.connect(self.db_path)
                c = db_bg.cursor()
                c.execute(
                    "INSERT OR REPLACE INTO matrix (id, bundle_dir, manifest_path, updated_ts) VALUES (1, ?, ?, ?)",
                    (self.weights_dir, self.weights_manifest_path, save_ts)
                )
                db_bg.commit()
                db_bg.close()
                self.last_weight_save_time = save_ts
            except Exception:
                pass

    def _log_episode(self, reward=0.0):
        self._queue_episode(reward=reward)

    def save_synapses(self, force=False, blocking=False):
        if force or blocking:
            try:
                self._background_save()
            except Exception:
                pass
            return
        if not self.is_alive:
            return
        if self.weight_save_thread is not None and self.weight_save_thread.is_alive():
            return
        self.last_weight_save_time = time.time()
        self.weight_save_thread = threading.Thread(target=self._background_save, daemon=True)
        self.weight_save_thread.start()

    def log(self, text):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {text}"
        self.terminal_log.append(line)
        if len(self.terminal_log) > 200:
            self.terminal_log.pop(0)
        self._queue_journal("INFO", "runtime", text)

    def play_sound(self, hz_start, hz_end, duration_ms):
        self.vocal_queue.append((hz_start, hz_end, duration_ms))

    def _store_audio_prototype(self):
        if self.voice_smooth < 0.05:
            return
        proto = np.copy(self.audio_nerve[:32])
        if len(self.heard_prototypes) < 12:
            self.heard_prototypes.append(proto)
        else:
            idx = random.randrange(len(self.heard_prototypes))
            self.heard_prototypes[idx] = 0.85 * self.heard_prototypes[idx] + 0.15 * proto

    def _emit_proto_speech(self):
        if not self.heard_prototypes:
            return False
        proto = random.choice(self.heard_prototypes)
        base = 220 + int(np.argmax(proto) * 18)
        self.play_sound(base, base + 90, 120)
        self.play_sound(base + 40, base - 20, 90)
        self.play_sound(base + 10, base + 70, 140)
        return True

    def _apply_synaptic_decay(self):
        decay_rate = 0.99980
        self.W_state *= decay_rate
        self.W_assoc *= decay_rate
        self.W_motor *= decay_rate
        self.W_touch *= 0.99990
        self.W_audio *= 0.99992
        self.W_state = np.clip(self.W_state, -1.10, 1.10)
        self.W_assoc = np.clip(self.W_assoc, -1.10, 1.10)
        self.W_motor = np.clip(self.W_motor, -1.10, 1.10)

    def inject_reward(self, amount=1.0):
        lr = np.float32(0.004 * amount)
        combined_hidden = np.concatenate([self.visual_hidden, self.audio_hidden, self.touch_hidden, self.vocal_hidden, self.associative_hidden])
        state_vec = np.concatenate([self.semantic_vision, self.drives.as_vector()])
        self.W_state += lr * np.outer(state_vec, self.associative_hidden)
        self.W_assoc += lr * np.outer(combined_hidden, self.associative_hidden)
        self.W_motor += lr * np.outer(self.associative_hidden, self.output_layer)
        self._apply_synaptic_decay()
        self.drives.step(0.0, self.semantic_vision, self.audio_volume, self.touch_nerve, 0.0, on_reward=True)
        self.internal_state = np.array([self.drives.energy, self.drives.stress, self.drives.curiosity], dtype=np.float32)
        self.log(f"[TRAINING] DOPAMINE INJECTED! (+{amount})")
        if amount > 0.5:
            self.play_sound(600, 800, 150)
        self.reward_flash = amount
        self._log_episode(reward=amount)

    def inject_punishment(self):
        lr = np.float32(0.004)
        combined_hidden = np.concatenate([self.visual_hidden, self.audio_hidden, self.touch_hidden, self.vocal_hidden, self.associative_hidden])
        state_vec = np.concatenate([self.semantic_vision, self.drives.as_vector()])
        self.W_state -= lr * np.outer(state_vec, self.associative_hidden)
        self.W_assoc -= lr * np.outer(combined_hidden, self.associative_hidden)
        self.W_motor -= lr * np.outer(self.associative_hidden, self.output_layer)
        self.W_motor += (np.random.randn(*self.W_motor.shape) * 0.01).astype(np.float32)
        self._apply_synaptic_decay()
        self.drives.step(0.0, self.semantic_vision, self.audio_volume, self.touch_nerve, 0.0, on_pain=True)
        self.internal_state = np.array([self.drives.energy, self.drives.stress, self.drives.curiosity], dtype=np.float32)
        self.log("[TRAINING] CORTISOL INJECTED! Current behavior scrambled.")
        self.play_sound(300, 100, 200)
        self._log_episode(reward=-1.0)

    def _run_audio_out(self):
        try:
            import winsound
            while self.is_alive:
                if self.vocal_queue:
                    hz_start, hz_end, duration_ms = self.vocal_queue.pop(0)
                    steps = max(1, duration_ms // 30)
                    for i in range(steps):
                        t = i / max(1, steps - 1)
                        freq = int(hz_start + (hz_end - hz_start) * t)
                        freq = max(37, min(32767, freq))
                        winsound.Beep(freq, max(10, duration_ms // steps))
                else:
                    time.sleep(0.01)
        except Exception:
            pass

    def _emit_behavior_vocal(self):
        now = time.time()
        if now - self.last_vocal_time < 2.0:
            return
        self.last_vocal_time = now

        if self.behavior_state == "seeking_social":
            self.play_sound(520, 760, 180)
        elif self.behavior_state == "investigating":
            self.play_sound(420, 520, 120)
        elif self.behavior_state == "startled":
            self.play_sound(700, 320, 140)
        elif self.behavior_state == "resting":
            self.play_sound(260, 280, 90)
        elif self.behavior_state == "settling":
            self.play_sound(300, 360, 120)
        elif self.behavior_state == "companioning":
            if not self._emit_proto_speech():
                self.play_sound(420, 640, 220)

    def _run_optics(self):
        cap = None
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)

            while self.is_alive:
                try:
                    now = time.time()
                    if now - self.last_desktop_capture_time >= self.desktop_capture_interval:
                        self._capture_desktop_eye()
                        self.last_desktop_capture_time = now
                    if now - self.last_panel_render_time >= self.panel_render_interval:
                        self._render_panel_eye()
                        self._render_control_line()
                        self.last_panel_render_time = now

                    webcam_frame = None
                    if cap is not None:
                        ret, frame = cap.read()
                        if ret:
                            webcam_frame = cv2.flip(cv2.resize(frame, (640, 480)), 1)

                    if self.embodiment == "REAL_WORLD":
                        self.camera_status = f"REAL_WORLD ({self.active_eye_source})"

                        source_frame = webcam_frame if webcam_frame is not None else self.raw_frame
                        if self.active_eye_source == "DESKTOP":
                            source_frame = cv2.resize(self.desktop_eye_frame.copy(), (640, 480), interpolation=cv2.INTER_LINEAR)
                        elif self.active_eye_source == "PANEL":
                            source_frame = cv2.resize(self.panel_eye_frame.copy(), (640, 480), interpolation=cv2.INTER_LINEAR)
                        elif self.active_eye_source == "FUTURE_BODY":
                            source_frame = cv2.resize(self.future_body_frame.copy(), (640, 480), interpolation=cv2.INTER_LINEAR)

                        if self.active_eye_source != "PANEL":
                            source_frame = self._apply_gaze_crosshair_to_frame(source_frame)
                        brain_frame = source_frame.copy() if self.active_eye_source == "PANEL" else self._apply_control_line_to_frame(source_frame)
                        optic, sem, faces, uppers, fovea, periph, _motion_map = self.vision.process_real_world(brain_frame, self.pan_pos, self.tilt_pos)
                        self.optic_nerve = optic
                        self.semantic_vision = sem
                        self.detected_faces = faces + uppers
                        self.fovea_frame = fovea
                        self.periphery_frame = periph
                        self.raw_frame = brain_frame.copy()

                        if sem[0] > 0.5:
                            self.last_face_seen_time = time.time()
                        self.face_presence_smooth = 0.90 * self.face_presence_smooth + 0.10 * sem[0]
                        self.face_lock_smooth = 0.90 * self.face_lock_smooth + 0.10 * sem[7]
                        self.motion_smooth = 0.90 * self.motion_smooth + 0.10 * sem[5]

                        if self.active_eye_source == "WEBCAM" and webcam_frame is not None:
                            self._apply_real_world_touch_inference(webcam_frame.shape)
                        else:
                            self.touch_nerve = np.zeros_like(self.touch_nerve)

                    else:
                        self.camera_status = f"DREAM_STATE ({self.active_eye_source})"
                        sim_render = self.sim_world.render_vision()
                        self.ego_frame = self.sim_world.get_ego_vision()

                        self.raw_frame = self._apply_control_line_to_frame(sim_render.copy())

                        if self.active_eye_source == "DESKTOP":
                            source_frame = cv2.resize(self.desktop_eye_frame.copy(), (640, 480), interpolation=cv2.INTER_LINEAR)
                            source_frame = self._apply_gaze_crosshair_to_frame(source_frame)
                            source_frame = self._apply_control_line_to_frame(source_frame)
                            optic, sem, faces, uppers, fovea, periph, _motion_map = self.vision.process_real_world(source_frame, self.pan_pos, self.tilt_pos)
                            self.detected_faces = faces + uppers
                            self.touch_nerve = np.zeros_like(self.touch_nerve)
                        elif self.active_eye_source == "PANEL":
                            source_frame = cv2.resize(self.panel_eye_frame.copy(), (640, 480), interpolation=cv2.INTER_LINEAR)
                            optic, sem, faces, uppers, fovea, periph, _motion_map = self.vision.process_real_world(source_frame, self.pan_pos, self.tilt_pos)
                            self.detected_faces = faces + uppers
                            self.touch_nerve = np.zeros_like(self.touch_nerve)
                        elif self.active_eye_source == "FUTURE_BODY":
                            source_frame = cv2.resize(self.future_body_frame.copy(), (640, 480), interpolation=cv2.INTER_LINEAR)
                            source_frame = self._apply_gaze_crosshair_to_frame(source_frame)
                            source_frame = self._apply_control_line_to_frame(source_frame)
                            optic, sem, faces, uppers, fovea, periph, _motion_map = self.vision.process_real_world(source_frame, self.pan_pos, self.tilt_pos)
                            self.detected_faces = faces + uppers
                            self.touch_nerve = np.zeros_like(self.touch_nerve)
                        else:
                            sim_social = self.sim_world.get_social_semantics()
                            dream_brain = self._apply_control_line_to_frame(self.ego_frame)
                            optic, sem, fovea, periph, _motion_map = self.vision.process_dream_state(dream_brain, self.pan_pos, self.tilt_pos, sim_social=sim_social)
                            self.detected_faces = []
                            self.touch_nerve = self.sim_world.get_touch_sensors()

                        self.optic_nerve = optic
                        self.semantic_vision = sem
                        self.fovea_frame = fovea
                        self.periphery_frame = periph
                        self.face_presence_smooth = 0.90 * self.face_presence_smooth + 0.10 * sem[0]
                        self.face_lock_smooth = 0.90 * self.face_lock_smooth + 0.10 * sem[7]
                        self.motion_smooth = 0.90 * self.motion_smooth + 0.10 * sem[5]

                    time.sleep(0.033)
                except Exception as exc:
                    self.camera_status = "OFFLINE/ERROR"
                    time.sleep(0.10)
        except Exception as exc:
            self.camera_status = "OFFLINE/ERROR"
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

    def _run_audio_in(self):
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.audio_sample_rate, input=True, frames_per_buffer=self.audio_fft_size)
            edges = np.geomspace(40.0, max(200.0, self.audio_sample_rate / 2.0), self.audio_size + 1)
            edge_bins = np.clip((edges / (self.audio_sample_rate / self.audio_fft_size)).astype(np.int32), 1, self.audio_fft_size // 2)
            band_centers = []
            for i in range(self.audio_size):
                band_centers.append(float((edges[i] + edges[i + 1]) * 0.5))
            self.audio_band_centers = np.array(band_centers, dtype=np.float32)

            while self.is_alive:
                data_raw = stream.read(self.audio_fft_size, exception_on_overflow=False)
                data = np.frombuffer(data_raw, dtype=np.int16).astype(np.float32)
                if data.size == 0:
                    time.sleep(0.01)
                    continue
                windowed = data * np.hanning(len(data))
                fft_out = np.abs(np.fft.rfft(windowed))[: self.audio_fft_size // 2]
                fft_log = np.log1p(fft_out)

                bands = np.zeros(self.audio_size, dtype=np.float32)
                for i in range(self.audio_size):
                    b0 = int(edge_bins[i])
                    b1 = int(max(edge_bins[i + 1], b0 + 1))
                    bands[i] = float(np.mean(fft_log[b0:b1]))

                frame_ref = float(np.percentile(bands, 95)) if np.max(bands) > 0 else 1.0
                self.audio_gain_reference = max(0.10, 0.995 * self.audio_gain_reference + 0.005 * frame_ref)
                scaled = np.clip(bands / (self.audio_gain_reference * 1.8 + 1e-6), 0.0, 1.0)
                self.audio_nerve = scaled.astype(np.float32)

                if self.audio_size >= self.audio_display_bands:
                    display = self.audio_nerve.reshape(self.audio_display_bands, -1).mean(axis=1)
                else:
                    display = np.interp(
                        np.linspace(0, self.audio_size - 1, self.audio_display_bands),
                        np.arange(self.audio_size),
                        self.audio_nerve,
                    )
                self.audio_display = np.maximum(display.astype(np.float32), self.audio_display * 0.84)

                rms = float(np.sqrt(np.mean(np.square(data / 32768.0))))
                self.audio_volume = float(np.clip(rms * 8.0, 0.0, 1.0))
                self.audio_smooth = 0.90 * self.audio_smooth + 0.10 * self.audio_volume

                voice_mask = (self.audio_band_centers >= 85.0) & (self.audio_band_centers <= 3400.0)
                voice_energy = float(np.mean(self.audio_nerve[voice_mask])) if np.any(voice_mask) else self.audio_volume
                self.voice_smooth = 0.92 * self.voice_smooth + 0.08 * voice_energy
                if voice_energy > 0.05 and random.random() < 0.10:
                    self._store_audio_prototype()
                time.sleep(0.005)
        except Exception:
            self.audio_nerve = np.zeros(self.audio_size, dtype=np.float32)
            self.audio_display = np.zeros(self.audio_display_bands, dtype=np.float32)
            self.audio_volume = 0.0


    def _update_sim_cues(self):
        if self.embodiment != "DREAM_STATE":
            self.sim_treat_signal = 0.0
            self.sim_treat_angle = 0.0
            self.sim_charge_signal = 0.0
            self.sim_charge_angle = 0.0
            self.sim_open_space = 1.0
            return

        def nearest_rel(items, radius=18.0):
            best = None
            heading = math.radians(self.sim_world.theta)
            for item in items:
                dx = float(item["x"] - self.sim_world.x)
                dy = float(item["y"] - self.sim_world.y)
                dist = math.hypot(dx, dy)
                ang = math.atan2(dy, dx)
                rel = (ang - heading + math.pi) % (2 * math.pi) - math.pi
                if best is None or dist < best[0]:
                    best = (dist, rel)
            if best is None:
                return 0.0, 0.0
            dist, rel = best
            strength = float(np.clip(1.0 - dist / self.sim_world.max_depth, 0.0, 1.0))
            return strength, float(np.clip(rel / max(self.sim_world.fov / 2.0, 1e-6), -1.0, 1.0))

        treat_strength, treat_angle = nearest_rel(self.sim_world.treats)
        charge_strength, charge_angle = nearest_rel([self.sim_world.charger])
        self.sim_treat_signal = 0.88 * self.sim_treat_signal + 0.12 * treat_strength
        self.sim_treat_angle = 0.70 * self.sim_treat_angle + 0.30 * treat_angle
        self.sim_charge_signal = 0.88 * self.sim_charge_signal + 0.12 * charge_strength
        self.sim_charge_angle = 0.70 * self.sim_charge_angle + 0.30 * charge_angle
        self.sim_open_space = float(np.clip(self.touch_nerve[2], 0.0, 1.0))

    def _update_presence_state(self, dt):
        social_seen = self.face_presence_smooth > 0.35
        heard_voice = self.voice_smooth > 0.035
        if social_seen:
            self.presence_hold = min(30.0, self.presence_hold + dt)
            self.absence_hold = max(0.0, self.absence_hold - dt * 2.0)
        else:
            self.presence_hold = max(0.0, self.presence_hold - dt * 0.8)
            self.absence_hold = min(30.0, self.absence_hold + dt)

        self.greet_cooldown = max(0.0, self.greet_cooldown - dt)

        if social_seen and self.presence_hold > 1.2 and self.greet_cooldown <= 0.0:
            self.log("[COMPANION] Stable social presence detected.")
            self.play_sound(500, 780, 260)
            self.greet_cooldown = 18.0

        if social_seen and heard_voice:
            self.companion_timer = min(20.0, self.companion_timer + dt)
        else:
            self.companion_timer = max(0.0, self.companion_timer - dt * 0.8)

    def _apply_idle_search(self, dt):
        if self.embodiment != "REAL_WORLD":
            return
        if self.face_presence_smooth < 0.18 and self.behavior_state in ("resting", "settling", "companioning"):
            self.desired_output[4] = np.tanh(self.desired_output[4] * 0.35)
            self.desired_output[5] = np.tanh(self.desired_output[5] * 0.35)
            return
        if self.face_presence_smooth > 0.20 or self.behavior_state not in ("roaming", "investigating", "seeking_social"):
            return
        self.idle_scan_phase += dt * (0.18 + 0.12 * self.drives.seek_drive)
        scan_x = math.sin(self.idle_scan_phase) * 0.28
        scan_y = math.sin(self.idle_scan_phase * 0.47) * 0.10
        self.desired_output[4] = np.tanh(self.desired_output[4] * 0.72 + scan_x * 0.28)
        self.desired_output[5] = np.tanh(self.desired_output[5] * 0.82 + scan_y * 0.18)


    def _update_gaze_controller(self, dt):
        social_seen = self.face_presence_smooth > 0.50
        if social_seen:
            tx = float(np.clip(self.semantic_vision[1], -1.0, 1.0))
            ty = float(np.clip(self.semantic_vision[2], -1.0, 1.0))
            target_changed = abs(tx - self.gaze_target_x) + abs(ty - self.gaze_target_y) > 0.18
            if self.gaze_mode != "pursuit" or target_changed or self.saccade_timer <= 0.0:
                self.gaze_mode = "saccade"
                self.gaze_target_x = tx
                self.gaze_target_y = ty
                self.saccade_timer = 0.10
                self.fixation_timer = 0.45
            else:
                self.gaze_mode = "pursuit"
                self.gaze_target_x = 0.85 * self.gaze_target_x + 0.15 * tx
                self.gaze_target_y = 0.85 * self.gaze_target_y + 0.15 * ty
        else:
            if self.behavior_state in ("resting", "settling", "companioning"):
                self.gaze_mode = "center"
                self.gaze_target_x = 0.0
                self.gaze_target_y = 0.0
            else:
                self.gaze_mode = "search"
                self.microsaccade_phase += dt * 1.4
                self.gaze_target_x = math.sin(self.microsaccade_phase) * 0.16
                self.gaze_target_y = math.sin(self.microsaccade_phase * 0.63) * 0.06

        self.saccade_timer = max(0.0, self.saccade_timer - dt)
        self.fixation_timer = max(0.0, self.fixation_timer - dt)

        # REDUCED HARDCODING HERE:
        # Instead of directly overwriting desired_output, we gently mix it with the neural network's raw output.
        # This makes him feel alive—he looks where his neurons want to look, but has a bias toward faces.
        
        base_pan = float(self.desired_output[4])
        base_tilt = float(self.desired_output[5])

        if self.gaze_mode == "saccade":
            self.desired_output[4] = np.clip(base_pan * 0.4 + self.gaze_target_x * 0.95, -0.85, 0.85)
            self.desired_output[5] = np.clip(base_tilt * 0.4 + self.gaze_target_y * 0.95, -0.85, 0.85)
            if self.saccade_timer <= 0.0:
                self.gaze_mode = "pursuit" if social_seen else "center"
        elif self.gaze_mode == "pursuit":
            self.desired_output[4] = np.tanh(base_pan * 0.55 + self.gaze_target_x * 0.45)
            self.desired_output[5] = np.tanh(base_tilt * 0.55 + self.gaze_target_y * 0.45)
        elif self.gaze_mode == "center":
            self.desired_output[4] = np.tanh(base_pan * 0.8)
            self.desired_output[5] = np.tanh(base_tilt * 0.8)
        elif self.gaze_mode == "search":
            self.desired_output[4] = np.tanh(base_pan * 0.70 + self.gaze_target_x * 0.30)
            self.desired_output[5] = np.tanh(base_tilt * 0.70 + self.gaze_target_y * 0.30)

    def _update_visit_memory(self, dt):
        if self.embodiment != "DREAM_STATE":
            return
        pos = (float(self.sim_world.x), float(self.sim_world.y))
        self.visit_history.append(pos)
        if len(self.visit_history) > 120:
            self.visit_history.pop(0)
        if len(self.visit_history) >= 30:
            recent = self.visit_history[-30:]
            cx = sum(p[0] for p in recent) / len(recent)
            cy = sum(p[1] for p in recent) / len(recent)
            spread = sum(math.hypot(p[0] - cx, p[1] - cy) for p in recent) / len(recent)
            self.loop_score = 0.92 * self.loop_score + 0.08 * (1.0 if spread < 48.0 else 0.0)
        else:
            self.loop_score *= 0.96

    def _micro_rehearsal(self, dt):
        if self.embodiment != "REAL_WORLD":
            return
        if self.absence_hold < 8.0:
            return
        rehearsal = 0.00025 * dt * (0.3 + self.drives.curiosity + self.drives.social_bond)
        state_vec = np.concatenate([self.semantic_vision, self.drives.as_vector()])
        self.W_state += rehearsal * np.outer(state_vec, self.associative_hidden)
        self.W_motor += (rehearsal * 0.35) * np.outer(self.associative_hidden, self.output_layer)
        self._apply_synaptic_decay()

    def bump_really_active(self):
        return float(self.touch_nerve[3]) > 0.5 or (float(self.touch_nerve[0]) + float(self.touch_nerve[1])) > 1.2

    def _update_stuck_state(self, dt, movement_mag):
        moved = math.hypot(self.virtual_body_x - self.last_body_x, self.virtual_body_y - self.last_body_y)
        trying_to_move = abs(float(self.output_layer[0])) + abs(float(self.output_layer[1])) > 0.30
        obstacle_pressure = (1.0 - float(self.touch_nerve[2])) + float(self.touch_nerve[3]) + max(float(self.touch_nerve[0]), float(self.touch_nerve[1]))
        if trying_to_move and movement_mag > 0.15 and moved < 0.35 and obstacle_pressure > 0.45:
            self.stuck_timer += dt
        else:
            self.stuck_timer = max(0.0, self.stuck_timer - dt * 1.5)
        self.last_body_x = self.virtual_body_x
        self.last_body_y = self.virtual_body_y
        if self.stuck_timer > 1.25:
            self.escape_turn_sign = -1.0 if float(self.touch_nerve[0]) > float(self.touch_nerve[1]) else 1.0
            self.behavior_state = "investigating"
            self.behavior_timer = 0.8
            self.stuck_timer = 0.4
            self.log("[STATE] Unsticking behavior engaged.")

    def tick(self, dt):
        if not self.is_alive:
            return
        self.reward_flash = max(0.0, self.reward_flash - 1.5 * dt)

        if self.is_booting:
            if time.time() - self.boot_timer > 0.6:
                self.boot_timer = time.time()
                if self.boot_step == 0:
                    self.log("[SYS] Initiating Autonomous Core Systems...")
                    self.play_sound(150, 150, 200)
                elif self.boot_step == 1:
                    self.log(f"[SYS] Checking Vision Cortex ...... {self.camera_status}")
                    self.play_sound(300, 300, 150)
                elif self.boot_step == 2:
                    self.log("[SYS] Checking Auditory Cortex .... ONLINE")
                    self.play_sound(450, 450, 150)
                elif self.boot_step == 3:
                    self.log("[SYS] Checking Tactile Bus ....... READY")
                    self.play_sound(600, 600, 150)
                elif self.boot_step == 4:
                    self.log(f"[BOOT COMPLETE] Engaging {self.total_hidden} Neural Pathways.")
                    self.play_sound(800, 800, 400)
                    self.is_booting = False
                self.boot_step += 1
            return

        self._update_presence_state(dt)
        self._update_visit_memory(dt)
        self._update_sim_cues()
        self._update_autonomous_attention(dt)
        self._update_panel_dwell_select(dt)

        self.visual_hidden = np.tanh(np.dot(self.optic_nerve, self.W_visual))
        self.audio_hidden = np.tanh(np.dot(self.audio_nerve, self.W_audio))
        self.touch_hidden = np.tanh(np.dot(self.touch_nerve, self.W_touch))

        drive_vec = self.drives.as_vector()
        vocal_in = np.concatenate([self.audio_nerve, self.semantic_vision, drive_vec]).astype(np.float32)
        self.vocal_hidden = np.tanh(np.dot(vocal_in, self.W_vocal))
        state_vec = np.concatenate([self.semantic_vision, drive_vec])
        state_bias = np.dot(state_vec, self.W_state)

        combined_hidden = np.concatenate([self.visual_hidden, self.audio_hidden, self.touch_hidden, self.vocal_hidden, self.associative_hidden])
        assoc_noise = (np.random.randn(self.associative_neurons).astype(np.float32) * 0.00025)
        self.associative_hidden = np.tanh(np.dot(combined_hidden, self.W_assoc) + state_bias + assoc_noise)
        raw_output = np.tanh(np.dot(self.associative_hidden, self.W_motor))

        self.behavior_timer -= dt
        if self.behavior_timer <= 0.0:
            self._choose_behavior_state()
        self.desired_output = raw_output.copy()
        self._apply_behavior_bias()
        self._update_gaze_controller(dt)
        self._apply_idle_search(dt)
        self._apply_output_regulation()

        smoothing = 0.12 if self.behavior_state in ("resting", "investigating") else 0.18
        self.output_layer = (1.0 - smoothing) * self.output_layer + smoothing * self.desired_output

        eye_gain = 4.8 if self.gaze_mode == "saccade" else (2.6 if self.gaze_mode == "pursuit" else 1.8)
        self.pan_pos += (self.output_layer[4] - self.pan_pos) * eye_gain * dt
        self.tilt_pos += (self.output_layer[5] - self.tilt_pos) * eye_gain * dt

        target_v = float((self.output_layer[0] + self.output_layer[1]) * 24.0)
        target_w = float((self.output_layer[1] - self.output_layer[0]) * 32.0)
        speed_blend = min(1.0, dt * 2.2)
        turn_blend = min(1.0, dt * 2.8)
        self.body_speed += (target_v - self.body_speed) * speed_blend
        self.body_turn_rate += (target_w - self.body_turn_rate) * turn_blend
        if abs(self.body_speed) < 0.45:
            self.body_speed = 0.0
        if abs(self.body_turn_rate) < 0.35:
            self.body_turn_rate = 0.0
        v = float(self.body_speed)
        w = float(self.body_turn_rate)
        self.last_cmd_v = v
        self.last_cmd_w = w
        movement_mag = abs(v) / 24.0 + abs(w) / 32.0
        self._update_stuck_state(dt, movement_mag)

        rad = math.radians(self.virtual_body_theta)
        self.virtual_body_x += v * math.cos(rad) * dt
        self.virtual_body_y += v * math.sin(rad) * dt
        self.virtual_body_theta = (self.virtual_body_theta + w * dt) % 360.0

        on_reward = False
        on_pain = False

        if self.embodiment == "DREAM_STATE":
            eaten, hit_wall, prox_warning, _on_charger_state = self.sim_world.step(v, w, dt)
            self.touch_nerve = self.sim_world.get_touch_sensors()

            if eaten:
                on_reward = True
                self.inject_reward()
                self.log("[SIM] Digital Apple consumed. Dopamine injected!")
            if hit_wall:
                on_pain = True
                self.inject_punishment()
                self.log("[SIM] Collision detected. Cortisol (Pain) injected!")
            elif prox_warning:
                lr = np.float32(0.0015 * dt)
                self.W_motor -= lr * np.outer(self.associative_hidden, self.output_layer)
                self._apply_synaptic_decay()

        self.drives.step(dt, self.semantic_vision, self.audio_volume, self.touch_nerve, movement_mag, on_reward, on_pain, on_charger=(self.touch_nerve[5] > 0.5 if len(self.touch_nerve) > 5 else False))
        self.internal_state = np.array([self.drives.energy, self.drives.stress, self.drives.curiosity], dtype=np.float32)

        reward = 0.0
        if self.face_presence_smooth > 0.4:
            reward += 0.0015 + 0.0025 * self.drives.social_bond
        if self.face_lock_smooth > 0.65:
            reward += 0.004
        if self.face_presence_smooth > 0.35 and self.audio_smooth > 0.08:
            reward += 0.004
        if self.motion_smooth > 0.06:
            reward += 0.001 * self.drives.curiosity
        if self.touch_nerve[5] > 0.5 and self.drives.energy < 0.50:
            reward += 0.008
        if self.touch_nerve[0] > 0.0 or self.touch_nerve[1] > 0.0:
            reward += 0.0007 * self.drives.curiosity

        if self.companion_timer > 2.5 and self.face_lock_smooth > 0.40:
            reward += 0.0035 + 0.002 * self.drives.social_bond

        if reward > 0.0:
            lr = np.float32(reward)
            combined_hidden = np.concatenate([self.visual_hidden, self.audio_hidden, self.touch_hidden, self.vocal_hidden, self.associative_hidden])
            state_vec = np.concatenate([self.semantic_vision, self.drives.as_vector()])
            self.W_state += lr * np.outer(state_vec, self.associative_hidden)
            self.W_assoc += (lr * 0.25) * np.outer(combined_hidden, self.associative_hidden)
            self.W_motor += (lr * 0.45) * np.outer(self.associative_hidden, self.output_layer)
            self._apply_synaptic_decay()
            self.reward_flash = min(1.0, self.reward_flash + reward * 8.0)

        if self.audio_volume > 0.40:
            lr = np.float32(0.0020 * dt)
            self.W_audio += lr * np.outer(self.audio_nerve, self.audio_hidden)
            self._apply_synaptic_decay()

        if np.max(self.touch_nerve[:4]) > 0.40:
            lr = np.float32(0.0024 * dt)
            self.W_touch += lr * np.outer(self.touch_nerve, self.touch_hidden)
            self._apply_synaptic_decay()

        self._micro_rehearsal(dt)

        vocal_tensor = self.output_layer[3]
        if vocal_tensor > 0.72 and random.random() < 0.03:
            self._emit_behavior_vocal()

        if time.time() - self.last_weight_save_time > self.weight_save_interval:
            self.save_synapses()
        if random.random() < 0.02 * dt:
            self._log_episode(reward=reward)
        self._flush_buffers(force=False)
        if time.time() - self.last_snapshot_time > self.snapshot_interval:
            self.create_snapshot(label="auto")

    def _choose_behavior_state(self):
        face_present = self.face_presence_smooth > 0.35
        touch_peak = float(max(self.touch_nerve[0], self.touch_nerve[1], self.touch_nerve[3], 1.0 - self.touch_nerve[2] if len(self.touch_nerve) > 2 else 0.0))
        low_energy = self.drives.energy < 0.28
        on_charger = bool(self.touch_nerve[5] > 0.5)
        treat_visible = self.sim_treat_signal > 0.22
        charge_visible = self.sim_charge_signal > 0.18

        if (touch_peak > 0.78 or self.drives.stress > 0.90) and self.bump_really_active():
            new_state = "startled"
            duration = random.uniform(0.35, 0.75)
        elif low_energy and on_charger:
            new_state = "resting"
            duration = random.uniform(2.4, 4.2)
        elif low_energy and self.embodiment == "DREAM_STATE" and charge_visible:
            new_state = "seeking_charge"
            duration = random.uniform(1.2, 2.4)
        elif self.embodiment == "DREAM_STATE" and self.drives.curiosity > 0.50 and treat_visible and self.drives.stress < 0.55:
            new_state = "foraging"
            duration = random.uniform(1.4, 2.8)
        elif self.companion_timer > 4.0 and face_present and self.drives.stress < 0.55:
            new_state = "companioning"
            duration = random.uniform(1.8, 3.4)
        elif self.presence_hold > 2.0 and face_present and self.drives.stress < 0.60:
            new_state = "settling"
            duration = random.uniform(1.2, 2.5)
        elif face_present:
            new_state = "seeking_social"
            duration = random.uniform(1.0, 2.2)
        elif self.loop_score > 0.55 or (self.motion_smooth > 0.08 or self.audio_smooth > 0.10):
            new_state = "investigating"
            duration = random.uniform(0.8, 1.7)
        elif self.drives.seek_drive > 0.50:
            new_state = "roaming"
            duration = random.uniform(1.6, 3.0)
        else:
            new_state = "resting"
            duration = random.uniform(1.2, 2.6)

        if new_state != self.behavior_state:
            self.behavior_state = new_state
            self.log(f"[STATE] Behavior -> {self.behavior_state.upper()}")
            if new_state == "roaming":
                self.roam_bias = random.uniform(-0.12, 0.12)
                self.roam_bias_timer = random.uniform(2.2, 4.5)
            if new_state in ("foraging", "seeking_charge"):
                self.roam_goal_timer = random.uniform(1.0, 2.0)
        self.behavior_timer = duration

    def _apply_behavior_bias(self):
        left_whisker, right_whisker, front_range, bump, imu, on_charger = self.touch_nerve
        face_x = self.semantic_vision[1]
        face_y = self.semantic_vision[2]
        obstacle_pressure = float(max(left_whisker, right_whisker, bump, 1.0 - front_range))

        def blend_drive(forward, turn, curiosity_boost=0.0, vocal_boost=0.0):
            self.desired_output[0] = np.tanh(forward - turn)
            self.desired_output[1] = np.tanh(forward + turn)
            self.desired_output[2] = np.tanh(self.desired_output[2] + curiosity_boost)
            self.desired_output[3] = np.tanh(self.desired_output[3] + vocal_boost)

        if self.behavior_state == "resting":
            self.desired_output[0] *= 0.18 if on_charger > 0.5 else 0.28
            self.desired_output[1] *= 0.18 if on_charger > 0.5 else 0.28
            self.desired_output[2] = np.tanh(self.desired_output[2] * 0.35)
            self.desired_output[3] *= 0.20
        elif self.behavior_state == "seeking_social":
            turn = np.clip(-face_x * (0.18 + 0.24 * self.drives.social_bond) + (left_whisker - right_whisker) * 0.16, -0.28, 0.28)
            closeness = np.clip((0.20 - self.semantic_vision[3]) * 2.0, -0.14, 0.18)
            forward = 0.10 + closeness - obstacle_pressure * 0.14
            blend_drive(forward, turn, 0.20, 0.10)
            self.desired_output[4] = np.tanh(self.desired_output[4] - 0.18 * face_x)
            self.desired_output[5] = np.tanh(self.desired_output[5] - 0.14 * face_y)
        elif self.behavior_state == "investigating":
            turn = np.clip((left_whisker - right_whisker) * 0.26 + self.escape_turn_sign * 0.06, -0.24, 0.24)
            forward = 0.08 if front_range > 0.28 and bump < 0.5 else -0.04
            blend_drive(forward, turn, 0.08, 0.02)
        elif self.behavior_state == "settling":
            self.desired_output[0] *= 0.38
            self.desired_output[1] *= 0.38
            self.desired_output[4] = np.tanh(self.desired_output[4] - 0.10 * face_x)
            self.desired_output[5] = np.tanh(self.desired_output[5] - 0.08 * face_y)
            self.desired_output[2] = np.tanh(self.desired_output[2] + 0.10)
            self.desired_output[3] = np.tanh(self.desired_output[3] + 0.04)
        elif self.behavior_state == "companioning":
            self.desired_output[0] *= 0.16
            self.desired_output[1] *= 0.16
            self.desired_output[4] = np.tanh(self.desired_output[4] - 0.06 * face_x)
            self.desired_output[5] = np.tanh(self.desired_output[5] - 0.05 * face_y)
            self.desired_output[2] = np.tanh(self.desired_output[2] + 0.14)
            self.desired_output[3] = np.tanh(self.desired_output[3] + 0.14)
        elif self.behavior_state == "seeking_charge":
            cue_turn = np.clip(self.sim_charge_angle * 0.26, -0.28, 0.28)
            avoid_turn = np.clip((left_whisker - right_whisker) * 0.20, -0.24, 0.24)
            turn = np.clip(cue_turn + avoid_turn, -0.30, 0.30)
            forward = 0.12 + self.sim_charge_signal * 0.14 - obstacle_pressure * 0.18
            if bump > 0.5:
                forward = -0.14
            blend_drive(forward, turn, 0.04, -0.02)
        elif self.behavior_state == "foraging":
            cue_turn = np.clip(self.sim_treat_angle * 0.24, -0.26, 0.26)
            avoid_turn = np.clip((left_whisker - right_whisker) * 0.16, -0.22, 0.22)
            turn = np.clip(cue_turn + avoid_turn, -0.28, 0.28)
            forward = 0.10 + self.sim_treat_signal * 0.10 - obstacle_pressure * 0.16
            blend_drive(forward, turn, 0.16, 0.05)
        elif self.behavior_state == "roaming":
            self.roam_goal_timer -= 0.033
            self.roam_bias_timer -= 0.033
            if self.roam_bias_timer <= 0.0:
                self.roam_bias = 0.82 * self.roam_bias + random.uniform(-0.08, 0.08)
                self.roam_bias = float(np.clip(self.roam_bias, -0.22, 0.22))
                self.roam_bias_timer = random.uniform(1.8, 4.2)
            if self.roam_goal_timer <= 0.0:
                if self.embodiment == "DREAM_STATE":
                    self.roam_heading = (self.sim_world.theta + random.uniform(-35.0, 35.0) + self.roam_bias * 80.0) % 360.0
                else:
                    self.roam_heading = random.uniform(0.0, 360.0)
                self.roam_goal_timer = random.uniform(2.0, 4.0)

            heading_err = 0.0
            if self.embodiment == "DREAM_STATE":
                heading_err = ((self.roam_heading - self.sim_world.theta + 180.0) % 360.0) - 180.0

            turn = np.clip((heading_err / 180.0) * 0.14 + self.roam_bias * 0.45 + (left_whisker - right_whisker) * 0.16 + self.loop_score * 0.10 * self.escape_turn_sign, -0.24, 0.24)
            forward = 0.18 + self.sim_open_space * 0.08 - obstacle_pressure * 0.16
            if self.loop_score > 0.55:
                forward *= 0.72
                turn += self.escape_turn_sign * 0.08
            blend_drive(forward, turn, 0.10, 0.00)
        elif self.behavior_state == "startled":
            reverse = -0.30 if bump > 0.5 or front_range < 0.18 else -0.14
            self.desired_output[0] = reverse
            self.desired_output[1] = reverse
            if left_whisker >= right_whisker:
                self.desired_output[4] = 0.65
            else:
                self.desired_output[4] = -0.65
            self.desired_output[2] = np.tanh(self.desired_output[2] + 0.26)
            self.desired_output[3] = np.tanh(self.desired_output[3] + 0.14)

        if bump > 0.5:
            self.desired_output[0] = min(self.desired_output[0], -0.28)
            self.desired_output[1] = min(self.desired_output[1], -0.28)

        damp = 0.48 + 0.52 * self.drives.energy
        if on_charger > 0.5:
            damp *= 0.55
        self.desired_output[0] *= damp
        self.desired_output[1] *= damp

    def _apply_real_world_touch_inference(self, frame_shape):
        if self.embodiment != "REAL_WORLD":
            return
        face_present = self.semantic_vision[0]
        face_x = self.semantic_vision[1]
        face_size = self.semantic_vision[3]
        motion = self.semantic_vision[5]

        left_whisker = 0.0
        right_whisker = 0.0
        if face_present > 0.45:
            side_strength = min(1.0, face_size * 7.5 + self.face_lock_smooth * 0.15)
            if face_x < -0.08:
                left_whisker = side_strength
            elif face_x > 0.08:
                right_whisker = side_strength

        front_range = 1.0 - min(1.0, face_size * 4.5) if face_present > 0.45 else 1.0
        bump = 1.0 if (motion > 0.60 and front_range < 0.08) else 0.0
        imu = float(np.clip(abs(self.output_layer[1] - self.output_layer[0]) * 0.5, 0.0, 1.0))

        self.touch_nerve = np.array([
            left_whisker,
            right_whisker,
            front_range,
            bump,
            imu,
            0.0
        ], dtype=np.float32)

    def _apply_output_regulation(self):
        self.desired_output[0] = np.clip(self.desired_output[0], -0.65, 0.65)
        self.desired_output[1] = np.clip(self.desired_output[1], -0.65, 0.65)
        self.desired_output[2] = np.clip(self.desired_output[2], -0.95, 0.95)
        self.desired_output[3] = np.clip(self.desired_output[3], -0.85, 0.85)
        self.desired_output[4] = np.clip(self.desired_output[4], -0.85, 0.85)
        self.desired_output[5] = np.clip(self.desired_output[5], -0.85, 0.85)

    def shutdown(self):
        if self.shutdown_started:
            return
        self.shutdown_started = True
        self.is_alive = False
        try:
            self.vocal_queue.clear()
        except Exception:
            pass

        try:
            self._flush_buffers(force=True)
        except Exception:
            pass

        try:
            age = time.time() - float(self.last_weight_save_time or 0.0)
            if age > 30.0:
                self.save_synapses(force=True, blocking=True)
                self._queue_journal("INFO", "runtime", "[MEMORY] Final weight save complete.")
            else:
                self._queue_journal("INFO", "runtime", "[MEMORY] Recent weight save reused on shutdown.")
        except Exception as exc:
            self.log(f"[MEMORY] Final weight save failed: {exc}")

        if getattr(self, "shutdown_snapshot_enabled", False):
            try:
                self.create_snapshot(label="shutdown")
            except Exception as exc:
                self.log(f"[SNAPSHOT] Shutdown snapshot failed: {exc}")

        try:
            self._flush_buffers(force=True)
        except Exception:
            pass

        try:
            if self.weight_save_thread is not None and self.weight_save_thread.is_alive():
                self.weight_save_thread.join(timeout=1.0)
        except Exception:
            pass

        try:
            with self.db_lock:
                self.db.close()
        except Exception:
            pass

        print("[SYSTEM] Cortex cleanly shutdown. Safe to close.")


# ==========================================
# THE UI 
# ==========================================

class ClickableVideoScreen(QLabel):
    def __init__(self, terminal):
        super().__init__()
        self.terminal = terminal
        self.setMouseTracking(True)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.pixmap() is not None:
            # Map UI click to the 640x480 resolution of the internal frame
            pm_size = self.pixmap().size()
            lbl_size = self.size()
            
            # KeepAspectRatio scaling offset
            x_off = (lbl_size.width() - pm_size.width()) / 2.0
            y_off = (lbl_size.height() - pm_size.height()) / 2.0
            
            click_x = event.position().x() - x_off
            click_y = event.position().y() - y_off
            
            if 0 <= click_x <= pm_size.width() and 0 <= click_y <= pm_size.height():
                nx = click_x / pm_size.width()
                ny = click_y / pm_size.height()
                real_x = int(nx * 640)
                real_y = int(ny * 480)
                self.terminal.cortex.process_manual_click(real_x, real_y)
        super().mousePressEvent(event)


class AudioEQBar(QWidget):
    def __init__(self, cortex):
        super().__init__()
        self.cortex = cortex
        self.setFixedHeight(60)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(8, 14, 22))
        if not self.cortex.is_alive or self.cortex.is_booting:
            return
        nerves = getattr(self.cortex, "audio_display", self.cortex.audio_nerve)
        if len(nerves) == 0:
            return
        bar_w = self.width() / len(nerves)
        h = self.height()
        theme = self.cortex.get_theme_colors()
        
        for i, vol in enumerate(nerves):
            v = float(np.clip(vol, 0.0, 1.0))
            bar_h = max(2, int(v * (h - 4)))
            color = QColor(*theme["text"]) if v > 0.72 else QColor(*theme["accent"])
            p.fillRect(int(i * bar_w), int(h - bar_h), max(1, int(bar_w - 2)), bar_h, color)
            if v > 0.85:
                p.fillRect(int(i * bar_w), int(h - bar_h), max(1, int(bar_w - 2)), 2, QColor(255, 255, 255))


class NeuralHeatmapGrid(QWidget):
    def __init__(self, cortex):
        super().__init__()
        self.cortex = cortex
        self.grid_w = 100
        self.grid_h = 100
        self.prev_img = np.zeros((self.grid_h, self.grid_w, 3), dtype=np.float32)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, event):
        if not self.cortex.is_alive:
            return

        neurons = np.concatenate([
            self.cortex.visual_hidden,
            self.cortex.audio_hidden,
            self.cortex.touch_hidden,
            self.cortex.vocal_hidden,
            self.cortex.associative_hidden
        ])
        padded = np.zeros(self.grid_h * self.grid_w, dtype=np.float32)
        padded[:len(neurons)] = neurons[: self.grid_h * self.grid_w]
        if self.cortex.is_booting:
            padded *= 0

        vals = np.clip(padded, -1.0, 1.0).reshape((self.grid_h, self.grid_w))
        img_arr = np.zeros((self.grid_h, self.grid_w, 3), dtype=np.float32)

        pos = np.clip(vals, 0.0, 1.0)
        neg = np.clip(-vals, 0.0, 1.0)
        magnitude = np.clip(np.abs(vals) * 1.5, 0.0, 1.0)

        img_arr[:, :, 1] = pos * 255.0
        img_arr[:, :, 0] = neg * 255.0
        img_arr[:, :, 2] = magnitude * 80.0

        pulse = min(1.0, float(self.cortex.audio_volume * 1.5 + self.cortex.semantic_vision[5] * 2.0 + self.cortex.touch_nerve[3] * 2.0))
        if pulse > 0.0:
            img_arr[:, :, 2] = np.clip(img_arr[:, :, 2] + pulse * 90.0, 0, 255)

        self.prev_img = self.prev_img * 0.55 + img_arr * 0.45
        out = np.ascontiguousarray(self.prev_img.astype(np.uint8))
        qimg = QImage(out.data, self.grid_w, self.grid_h, self.grid_w * 3, QImage.Format.Format_RGB888).copy()

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(10, 15, 25))
        p.drawImage(self.rect(), qimg)


class CaptainShellOverlay(QWidget):
    def __init__(self, cortex):
        flags = Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        super().__init__(None, flags)
        self.cortex = cortex
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        try:
            self.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus, True)
        except Exception:
            pass
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.resize(250, 200)
        self._anchor_margin = 18
        self._drag_offset = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)
        QTimer.singleShot(150, self.anchor_to_corner)
        self.pulse_phase = 0.0

    def _tick(self):
        if not self.cortex.is_alive:
            try:
                self.close()
            except Exception:
                pass
            return
        self.pulse_phase += 0.15 + self.cortex.audio_volume
        self.update()

    def anchor_to_corner(self):
        app = QApplication.instance()
        if app is None:
            return
        screen = app.primaryScreen()
        if screen is None:
            return
        rect = screen.availableGeometry()
        x = rect.x() + rect.width() - self.width() - self._anchor_margin
        y = rect.y() + rect.height() - self.height() - self._anchor_margin
        self.move(x, y)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def closeEvent(self, event):
        try:
            self.timer.stop()
        except Exception:
            pass
        event.accept()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), Qt.GlobalColor.transparent)

        theme = self.cortex.get_theme_colors()
        accent = QColor(*theme["accent"])
        accent2 = QColor(*theme["accent2"])
        warn = QColor(*theme["warn"])
        text_color = QColor(*theme["text"])
        
        cx = self.width() // 2
        cy = 80
        
        # Cyber-Core procedural 3D illusion
        
        # 1. Base glow
        glow_grad = QRadialGradient(cx, cy, 60)
        glow_grad.setColorAt(0, QColor(accent.red(), accent.green(), accent.blue(), 50))
        glow_grad.setColorAt(1, QColor(0, 0, 0, 0))
        p.setBrush(glow_grad)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(cx - 60, cy - 60, 120, 120)

        # 2. Main structural sphere
        p.setBrush(QColor(10, 15, 25, 200))
        p.setPen(QPen(accent, 2))
        p.drawEllipse(cx - 40, cy - 40, 80, 80)
        
        # 3. Rotating orbital rings (speed based on curiosity/energy)
        p.setBrush(Qt.BrushStyle.NoBrush)
        energy_spin = self.pulse_phase * (0.5 + self.cortex.drives.energy * 1.5)
        
        p.setPen(QPen(accent2, 1))
        a1 = int((energy_spin * 20) % 360)
        p.drawArc(cx - 45, cy - 45, 90, 90, a1 * 16, 120 * 16)
        p.drawArc(cx - 45, cy - 45, 90, 90, (a1 + 180) * 16, 120 * 16)
        
        a2 = int((-energy_spin * 15) % 360)
        p.drawArc(cx - 50, cy - 50, 100, 100, a2 * 16, 60 * 16)
        p.drawArc(cx - 50, cy - 50, 100, 100, (a2 + 180) * 16, 60 * 16)

        # 4. Central Iris (reacts to audio & pan/tilt)
        fx, fy = self.cortex._current_focus_vector()
        # Map pan/tilt to 3D sphere shift
        px = int(fx * 18)
        py = int(fy * 18)
        
        vol_pulse = int(self.cortex.audio_volume * 20)
        iris_rad = 12 + vol_pulse
        
        # Determine color of iris (red if stressed)
        iris_color = warn if self.cortex.drives.stress > 0.75 else accent2
        
        p.setBrush(iris_color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(cx + px - iris_rad//2, cy + py - iris_rad//2, iris_rad, iris_rad)
        
        # Inner pupil
        p.setBrush(QColor(0,0,0))
        p.drawEllipse(cx + px - 3, cy + py - 3, 6, 6)
        
        # Highlight reflection
        p.setBrush(QColor(255, 255, 255, 150))
        p.drawEllipse(cx + px - iris_rad//2 + 2, cy + py - iris_rad//2 + 2, 4, 4)

        # --- TERMINAL LOG TEXT (Underneath core) ---
        p.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        lines = self.cortex.terminal_log[-3:] if self.cortex.terminal_log else ["Captain core online."]
        text_y = cy + 70
        
        for line in lines:
            msg = line[-45:] # Truncate so it fits width
            # Text shadow for desktop readability
            p.setPen(QColor(0, 0, 0, 200))
            p.drawText(1, text_y + 1, self.width(), 16, int(Qt.AlignmentFlag.AlignCenter), msg)
            
            p.setPen(text_color)
            p.drawText(0, text_y, self.width(), 16, int(Qt.AlignmentFlag.AlignCenter), msg)
            text_y += 14


class CommandTerminal(QMainWindow):
    def __init__(self, cortex):
        super().__init__()
        self.cortex = cortex
        self.setWindowTitle("Captain AI | Companion")
        self.setMinimumSize(1400, 850)
        self.setStyleSheet("background-color: #080c14; color: #00ffcc; font-family: 'Consolas', monospace;")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        grid = QGridLayout(main_widget)
        grid.setContentsMargins(15, 15, 15, 15)
        grid.setSpacing(15)

        opt_frame = QFrame()
        opt_frame.setStyleSheet("border: 2px solid #1a2a40; background: #0b101a; border-radius: 8px;")
        opt_layout = QVBoxLayout(opt_frame)
        opt_layout.setContentsMargins(0, 0, 0, 0)
        self.cam_title = QLabel(" <b style='color:#00ffff; font-size: 18px;'>[ OPTICS ] REAL WORLD</b>")
        self.cam_title.setStyleSheet("border: none; border-bottom: 2px solid #1a2a40; padding: 12px; background: #0d1424;")
        opt_layout.addWidget(self.cam_title)

        # Replaced standard QLabel with custom ClickableVideoScreen
        self.cam_label = ClickableVideoScreen(self)
        self.cam_label.setFont(QFont("Consolas", 24, QFont.Weight.Bold))
        self.cam_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.cam_label.setMinimumSize(640, 480)
        self.cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        opt_layout.addWidget(self.cam_label, 1)
        grid.addWidget(opt_frame, 0, 0, 3, 1)

        heat_frame = QFrame()
        heat_frame.setStyleSheet("border: 2px solid #1a2a40; background: #0b101a; border-radius: 8px;")
        heat_layout = QVBoxLayout(heat_frame)
        heat_layout.setContentsMargins(0, 0, 0, 0)
        hlbl = QLabel(f" <b style='color:#00ff66; font-size: 18px;'>[ HIDDEN CORTEX ] {self.cortex.total_hidden} NEURONS</b>")
        hlbl.setStyleSheet("border: none; border-bottom: 2px solid #1a2a40; padding: 12px; background: #0d1424;")
        heat_layout.addWidget(hlbl)
        self.heatmap = NeuralHeatmapGrid(cortex)
        heat_layout.addWidget(self.heatmap, 1)

        albl = QLabel(f" <b style='color:#00ccff; font-size: 14px;'>[ {self.cortex.audio_display_bands}-BAND AUDITORY CORTEX | V32 ]</b>")
        albl.setStyleSheet("border: none; border-top: 2px solid #1a2a40; padding: 5px;")
        heat_layout.addWidget(albl)
        self.audio_eq = AudioEQBar(cortex)
        heat_layout.addWidget(self.audio_eq)
        grid.addWidget(heat_frame, 0, 1, 1, 1)

        out_frame = QFrame()
        out_frame.setStyleSheet("border: 2px solid #1a2a40; background: #0b101a; border-radius: 8px;")
        out_layout = QHBoxLayout(out_frame)
        out_layout.setContentsMargins(15, 15, 15, 15)

        self.data_label = QLabel()
        self.data_label.setStyleSheet("color: #00ffff; font-size: 16px; line-height: 1.5; border: none;")
        out_layout.addWidget(self.data_label, 1)

        training_lbl = QLabel(
            "<b style='color:#ffcc00; font-size: 18px;'>[ TRAINING & MODES ]</b><br><br>"
            "<b style='color:#00ff66;'>UP ARROW</b>: Inject Dopamine (Reward)<br>"
            "<b style='color:#ff3333;'>DOWN ARROW</b>: Inject Cortisol (Punish)<br>"
            "<b style='color:#66ccff;'>1</b>: Webcam &nbsp;&nbsp; "
            "<b style='color:#66ccff;'>2</b>: Desktop &nbsp;&nbsp; "
            "<b style='color:#66ccff;'>3</b>: Panel &nbsp;&nbsp; "
            "<b style='color:#66ccff;'>4</b>: Dream<br><br>"
            "Watch the crosshair on the active eye.<br><b style='color:#66ccff;'>A</b>: Toggle AUTO attention.<br><b style='color:#66ccff;'>LEFT / RIGHT</b>: Cycle shell theme.<br>Captain always sees his control line. Dwell can select buttons."
        )
        training_lbl.setStyleSheet("color: #bbbbbb; font-size: 16px; line-height: 1.5; border: none;")
        out_layout.addWidget(training_lbl, 1)
        grid.addWidget(out_frame, 1, 1, 1, 1)

        term_frame = QFrame()
        term_frame.setStyleSheet("border: 2px solid #1a2a40; background: #05080c; border-radius: 8px;")
        term_layout = QVBoxLayout(term_frame)
        term_layout.setContentsMargins(0, 0, 0, 0)
        tlbl = QLabel(" <b style='color:#aaaaaa; font-size: 14px;'>[ SYSTEM LOG ] NEUROPLASTICITY EVENTS</b>")
        tlbl.setStyleSheet("border: none; border-bottom: 1px solid #1a2a40; padding: 5px; background: #0a0f18;")
        term_layout.addWidget(tlbl)

        self.term_text = QTextEdit()
        self.term_text.setReadOnly(True)
        self.term_text.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.term_text.setFixedHeight(90)
        self.term_text.setStyleSheet("border: none; color: #00ffcc; font-size: 13px; padding: 5px;")
        term_layout.addWidget(self.term_text)
        grid.addWidget(term_frame, 2, 1, 1, 1)

        grid.setColumnStretch(0, 5)
        grid.setColumnStretch(1, 4)
        grid.setRowStretch(0, 8)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(2, 1)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(33)

        QTimer.singleShot(500, self.cortex.start_hardware)

        self.shell_overlay = CaptainShellOverlay(self.cortex)
        self.shell_overlay.show()

    def closeEvent(self, event):
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "shell_overlay") and self.shell_overlay is not None:
                self.shell_overlay.close()
        except Exception:
            pass
        try:
            self.hide()
        except Exception:
            pass
        try:
            self.cortex.shutdown()
        except Exception:
            pass
        app = QApplication.instance()
        if app is not None:
            try:
                app.closeAllWindows()
            except Exception:
                pass
            QTimer.singleShot(0, app.quit)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Up:
            self.cortex.inject_reward()
        elif event.key() == Qt.Key.Key_Down:
            self.cortex.inject_punishment()
        elif event.key() == Qt.Key.Key_S:
            self.cortex.create_snapshot(label="manual")
        elif event.key() == Qt.Key.Key_Space:
            return
        elif event.key() == Qt.Key.Key_A:
            self.cortex._toggle_attention_mode()
        elif event.key() == Qt.Key.Key_Left:
            self.cortex._cycle_theme(-1, reason="manual")
        elif event.key() == Qt.Key.Key_Right:
            self.cortex._cycle_theme(1, reason="manual")
        elif event.key() == Qt.Key.Key_1:
            self.cortex._set_active_eye_source("WEBCAM", origin="manual")
        elif event.key() == Qt.Key.Key_2:
            self.cortex._set_active_eye_source("DESKTOP", origin="manual")
        elif event.key() == Qt.Key.Key_3:
            self.cortex._set_active_eye_source("PANEL", origin="manual")
        elif event.key() == Qt.Key.Key_4:
            self.cortex._set_active_eye_source("DREAM_STATE", origin="manual")

    def update_ui(self):
        self.cortex._capture_desktop_eye()
        
        theme = self.cortex.get_theme_colors()
        
        eye = getattr(self.cortex, "active_eye_source", "WEBCAM")
        if self.cortex.embodiment == "DREAM_STATE":
            self.cam_title.setText(f" <b style='color:rgb{theme['accent']}; font-size: 18px;'>[ OPTICS ] DREAM STATE | {eye} | + CONTROL LINE</b>")
        else:
            self.cam_title.setText(f" <b style='color:rgb{theme['text']}; font-size: 18px;'>[ OPTICS ] REAL WORLD | {eye} | + CONTROL LINE</b>")
        
        raw = self.cortex.raw_frame
        if (raw is None or raw.sum() == 0) and eye == "DESKTOP":
            raw = cv2.resize(self.cortex.desktop_eye_frame.copy(), (640, 480), interpolation=cv2.INTER_LINEAR)
            raw = self.cortex._apply_gaze_crosshair_to_frame(raw)
            raw = self.cortex._apply_control_line_to_frame(raw)
        elif (raw is None or raw.sum() == 0) and eye == "PANEL":
            raw = cv2.resize(self.cortex.panel_eye_frame.copy(), (640, 480), interpolation=cv2.INTER_LINEAR)
            
        if raw is not None and raw.sum() > 0:
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape

            if self.cortex.embodiment == "REAL_WORLD" and eye != "PANEL":
                if eye == "DESKTOP":
                    pass # Crosshair already drawn on desktop frame in backend to act as cursor
                else:
                    # Regular webcam crosshair
                    pan = self.cortex.pan_pos
                    tilt = self.cortex.tilt_pos
                    cx = int((pan + 1.0) / 2.0 * w)
                    cy = int((tilt + 1.0) / 2.0 * h)
                    cx = max(10, min(w - 10, cx))
                    cy = max(10, min(h - 10, cy))
                    accent_rgb = theme["accent"]
                    
                    # Target box style if hovering, otherwise cross
                    if self.cortex.control_line_hover_target:
                        cv2.rectangle(rgb, (cx-10, cy-10), (cx+10, cy+10), accent_rgb, 1)
                        cv2.circle(rgb, (cx, cy), 2, accent_rgb, -1)
                    else:
                        cv2.circle(rgb, (cx, cy), 12, accent_rgb, 1)
                        cv2.line(rgb, (cx - 7, cy - 7), (cx + 7, cy + 7), accent_rgb, 2)
                        cv2.line(rgb, (cx - 7, cy + 7), (cx + 7, cy - 7), accent_rgb, 2)
            else:
                if eye == "DREAM_STATE" and hasattr(self.cortex, 'ego_frame') and getattr(self.cortex, 'ego_frame') is not None:
                    try:
                        ego_rgb = cv2.cvtColor(self.cortex.ego_frame, cv2.COLOR_BGR2RGB)
                        ego_resized = cv2.resize(ego_rgb, (300, 150), interpolation=cv2.INTER_NEAREST)
                        eh, ew, _ = ego_resized.shape
                        ox, oy = w - ew - 20, 20
                        cv2.rectangle(rgb, (ox - 2, oy - 2), (ox + ew + 2, oy + eh + 2), theme["text"], 2)
                        rgb[oy:oy + eh, ox:ox + ew] = ego_resized
                        cv2.putText(rgb, "[ AI POV ]", (ox, oy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, theme["text"], 2)
                    except Exception:
                        pass


            if self.cortex.reward_flash > 0:
                overlay = np.full_like(rgb, (0, 255, 0))
                cv2.addWeighted(overlay, self.cortex.reward_flash * 0.3, rgb, 1.0, 0, rgb)

            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            self.cam_label.setPixmap(QPixmap.fromImage(qimg).scaled(
                self.cam_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        else:
            self.cam_label.setText(f"[{self.cortex.camera_status}]")

        self.heatmap.update()
        self.audio_eq.update()

        outs = self.cortex.output_layer
        dom_freq = float(self.cortex.audio_band_centers[int(np.argmax(self.cortex.audio_nerve))]) if np.sum(self.cortex.audio_nerve) > 0 else 0.0
        sim_data = f"SIM APPLES : {self.cortex.sim_world.apples_eaten}\n" if self.cortex.embodiment == "DREAM_STATE" else ""
        touch = self.cortex.touch_nerve
        self.data_label.setText(
            f"DOMINANT FREQ : {dom_freq:.0f} Hz\n"
            f"PEAK VOLUME   : {self.cortex.audio_volume:.4f}\n"
            f"ENERGY STATE  : {self.cortex.drives.energy:.4f}\n"
            f"STRESS STATE  : {self.cortex.drives.stress:.4f}\n"
            f"CURIOSITY     : {self.cortex.drives.curiosity:.4f}\n"
            f"BOND          : {self.cortex.drives.social_bond:.4f}\n"
            f"MOOD          : {self.cortex.drives.mood_string()}\n"
            f"BEHAVIOR      : {self.cortex.behavior_state.upper()}\n"
            f"AUTO TARGET   : {self.cortex.autonomous_eye_target}\n"
            f"THEME         : {self.cortex.avatar_theme.upper()}\n"
            f"CTRL HOVER    : {self.cortex.control_line_hover_target or self.cortex.panel_hover_target or 'NONE'}\n"
            f"FACE PRESENT  : {self.cortex.face_presence_smooth:.2f}\n"
            f"FACE LOCK     : {self.cortex.face_lock_smooth:.2f}\n"
            f"WHISKERS L/R  : {touch[0]:.2f} / {touch[1]:.2f}\n"
            f"BUMP / RANGE  : {touch[3]:.2f} / {touch[2]:.2f}\n"
            + sim_data + "\n"
            f"TENSOR[0] (L_MOTOR) : {outs[0]:+0.4f}\n"
            f"TENSOR[1] (R_MOTOR) : {outs[1]:+0.4f}\n"
            f"TENSOR[2] (FOCUS)   : {outs[2]:+0.4f}\n"
            f"TENSOR[3] (VOCAL)   : {outs[3]:+0.4f}\n"
            f"TENSOR[4] (PAN_POS) : {outs[4]:+0.4f}\n"
            f"TENSOR[5] (TLT_POS) : {outs[5]:+0.4f}\n\n"
            f"SYNAPSES   : {self.cortex.total_synapses:,}\nDB FILE    : {os.path.basename(self.cortex.db_path)}\nJOURNAL    : {os.path.basename(self.cortex.journal_path)}"
        )

        new_log = "\n".join(self.cortex.terminal_log)
        if self.term_text.toPlainText() != new_log:
            self.term_text.setPlainText(new_log)
            # Only scroll to bottom when a new entry is added, allowing users to scroll
            cursor = self.term_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.term_text.setTextCursor(cursor)


def brain_loop(cortex):
    last_time = time.time()
    while cortex.is_alive:
        dt = time.time() - last_time
        if dt >= 0.033:
            cortex.tick(dt)
            last_time = time.time()
        else:
            time.sleep(0.001)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    cortex = RawCortex()
    t = threading.Thread(target=brain_loop, args=(cortex,), daemon=True)
    t.start()

    window = CommandTerminal(cortex)
    window.show()

    exit_code = app.exec()
    if cortex.is_alive:
        try:
            cortex.shutdown()
        except Exception:
            pass
    sys.exit(exit_code)