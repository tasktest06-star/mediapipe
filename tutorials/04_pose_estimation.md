# Tutorial 04: Pose Estimation

## Introduction

Human pose estimation is the problem of detecting and localising the joints of the human body
in images or video. MediaPipe's BlazePose model provides 33 3D body landmarks at real-time
speeds on mobile and desktop hardware. Unlike earlier pose models that required cropped person
images, BlazePose runs end-to-end from a full frame — a major practical advantage.

Applications include:
- Fitness and sports coaching (rep counting, form correction)
- Physical therapy and rehabilitation monitoring
- Dance and animation capture
- Ergonomics and posture analysis

This tutorial builds two complete applications: a bicep-curl rep counter and a squat counter,
plus a reusable joint-angle utility used by both.

---

## 1. BlazePose: Model Overview

BlazePose returns 33 landmarks in a two-stage pipeline:
1. **Detector** — identifies the person in the frame with a bounding box and a few keypoints
   for alignment.
2. **Landmark regressor** — given the aligned crop, predicts all 33 landmarks with (x, y, z,
   visibility) for each.

The 33 landmarks cover the full body:

| Index range | Region |
|-------------|--------|
| 0 | NOSE |
| 1–10 | Face (eyes, ears, mouth) |
| 11–12 | Shoulders |
| 13–14 | Elbows |
| 15–16 | Wrists |
| 17–22 | Hands (finger tips) |
| 23–24 | Hips |
| 25–26 | Knees |
| 27–28 | Ankles |
| 29–32 | Feet (heels, toe tips) |

Each landmark has:
- `x, y` — normalised image coordinates [0,1]
- `z` — depth relative to hips (negative = closer to camera)
- `visibility` — confidence that the landmark is visible [0,1]

### Model selection

| `model_complexity` | Model | Best for |
|-------------------|-------|---------|
| 0 | Lite | Real-time apps, mobile |
| 1 | Full | Good balance (default) |
| 2 | Heavy | High-accuracy offline analysis |

---

## 2. Full-Body vs Upper-Body Mode

Before MediaPipe 0.9, there was a separate `upper_body_only` flag. In current versions, use
`model_complexity` to trade speed for accuracy. For upper-body-only applications (e.g.,
seated desk ergonomics), you can simply ignore landmarks 23 and above.

```python
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# Full body
pose_full = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5)

# Treat as upper-body only: ignore lower landmarks
UPPER_BODY_CONNECTIONS = [
    conn for conn in mp_pose.POSE_CONNECTIONS
    if max(conn) <= 22   # landmarks 0-22 are upper body
]
```

---

## 3. Joint Angle Utility

Both rep-counting applications rely on calculating the angle formed at a joint by three
landmark points.

```python
"""
Shared utility: compute the angle at joint B given points A, B, C.
"""

import numpy as np

def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) at point B formed by A-B-C.

    Parameters
    ----------
    a, b, c : array-like, shape (2,) or (3,)
        3D or 2D coordinates of the three points.
        B is the vertex of the angle.

    Returns
    -------
    float : angle in degrees [0, 180]
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

def get_coords(landmarks, idx):
    """Extract [x, y, z] from a landmark at the given index."""
    lm = landmarks.landmark[idx]
    return [lm.x, lm.y, lm.z]
```

---

## 4. Bicep-Curl Rep Counter

The curl counter measures the elbow angle:
- **Down position** — arm extended, angle > 160°
- **Up position** — arm curled, angle < 30°

A rep is counted each time the user moves from down to up and back to down.

```python
"""
Tutorial 04 – Bicep Curl Rep Counter
Counts reps for left and right arms independently.
"""

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def get_pt(landmarks, idx):
    lm = landmarks.landmark[idx]
    return [lm.x, lm.y]

class ArmCurlCounter:
    def __init__(self, angle_down=160, angle_up=30):
        self.count = 0
        self.stage = "down"   # 'up' or 'down'
        self.angle_down = angle_down
        self.angle_up = angle_up

    def update(self, angle):
        if angle > self.angle_down:
            self.stage = "down"
        if angle < self.angle_up and self.stage == "down":
            self.stage = "up"
            self.count += 1
        return self.count, self.stage

LEFT  = {"shoulder": 11, "elbow": 13, "wrist": 15}
RIGHT = {"shoulder": 12, "elbow": 14, "wrist": 16}

def draw_counter(frame, counter, label, x, y, angle):
    cv2.rectangle(frame, (x, y - 60), (x + 180, y), (20, 20, 20), -1)
    cv2.putText(frame, label, (x + 5, y - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, str(counter.count), (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 128), 2)
    cv2.putText(frame, counter.stage, (x + 70, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    cv2.putText(frame, f"{angle:.0f}deg", (x + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)

def main():
    cap = cv2.VideoCapture(0)
    left_counter  = ArmCurlCounter()
    right_counter = ArmCurlCounter()

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            rgb.flags.writeable = True

            if results.pose_landmarks:
                lms = results.pose_landmarks

                # Left arm
                lsh  = get_pt(lms, LEFT["shoulder"])
                lelb = get_pt(lms, LEFT["elbow"])
                lwr  = get_pt(lms, LEFT["wrist"])
                l_angle = calculate_angle(lsh, lelb, lwr)
                l_count, _ = left_counter.update(l_angle)

                # Right arm
                rsh  = get_pt(lms, RIGHT["shoulder"])
                relb = get_pt(lms, RIGHT["elbow"])
                rwr  = get_pt(lms, RIGHT["wrist"])
                r_angle = calculate_angle(rsh, relb, rwr)
                r_count, _ = right_counter.update(r_angle)

                # Draw skeleton
                mp_draw.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2,
                                        circle_radius=2),
                    mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2,
                                        circle_radius=2),
                )

                # Draw angle at elbow
                for elbow_idx, angle in [(13, l_angle), (14, r_angle)]:
                    ex = int(lms.landmark[elbow_idx].x * w)
                    ey = int(lms.landmark[elbow_idx].y * h)
                    cv2.putText(frame, f"{angle:.0f}", (ex - 20, ey - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)

                draw_counter(frame, left_counter,  "L ARM", 10,  80, l_angle)
                draw_counter(frame, right_counter, "R ARM", w-200, 80, r_angle)

            cv2.imshow("Bicep Curl Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

---

## 5. Squat Counter

The squat counter measures the knee angle:
- **Standing** — knee angle > 160°
- **Squat depth** — knee angle < 100°

```python
"""
Tutorial 04 – Squat Counter
Counts squats by monitoring bilateral knee angles.
"""

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba, bc = a - b, c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

def lm_xy(landmarks, idx):
    lm = landmarks.landmark[idx]
    return [lm.x, lm.y]

class SquatCounter:
    def __init__(self, angle_stand=160, angle_squat=100):
        self.count  = 0
        self.stage  = "up"
        self.angle_stand = angle_stand
        self.angle_squat = angle_squat

    def update(self, left_angle, right_angle):
        avg = (left_angle + right_angle) / 2
        if avg > self.angle_stand:
            self.stage = "up"
        if avg < self.angle_squat and self.stage == "up":
            self.stage = "down"
            self.count += 1
        return self.count, self.stage, avg

POSE_LMS = {
    "l_hip": 23, "l_knee": 25, "l_ankle": 27,
    "r_hip": 24, "r_knee": 26, "r_ankle": 28,
}

def main():
    cap = cv2.VideoCapture(0)
    counter = SquatCounter()

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lms = results.pose_landmarks
                la = calculate_angle(
                    lm_xy(lms, POSE_LMS["l_hip"]),
                    lm_xy(lms, POSE_LMS["l_knee"]),
                    lm_xy(lms, POSE_LMS["l_ankle"]),
                )
                ra = calculate_angle(
                    lm_xy(lms, POSE_LMS["r_hip"]),
                    lm_xy(lms, POSE_LMS["r_knee"]),
                    lm_xy(lms, POSE_LMS["r_ankle"]),
                )
                count, stage, avg = counter.update(la, ra)

                mp_draw.draw_landmarks(
                    frame, lms, mp_pose.POSE_CONNECTIONS)

                # HUD
                cv2.rectangle(frame, (0, 0), (280, 100), (20, 20, 20), -1)
                cv2.putText(frame, f"Reps: {count}", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
                cv2.putText(frame, f"Stage: {stage}", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 200, 0), 2)
                cv2.putText(frame, f"Knee: {avg:.0f}deg",
                            (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (100, 200, 255), 2)

            cv2.imshow("Squat Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

---

## 6. Real-Time Performance Optimisation

### 6.1 Frame skipping

If inference is slower than capture (e.g., on an older CPU), skip frames to reduce queue
build-up while still displaying every frame:

```python
inference_every_n = 2
frame_count = 0
cached_results = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % inference_every_n == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cached_results = pose.process(rgb)

    if cached_results and cached_results.pose_landmarks:
        mp_draw.draw_landmarks(frame, cached_results.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### 6.2 Threading

Decouple capture (I/O-bound) from inference (CPU-bound):

```python
import threading
import queue

frame_queue   = queue.Queue(maxsize=2)
result_queue  = queue.Queue(maxsize=2)

def capture_thread(cap, q):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not q.full():
            q.put(frame)

def inference_thread(pose, fq, rq):
    while True:
        if fq.empty():
            continue
        frame = fq.get()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if not rq.full():
            rq.put((frame, results))
```

### 6.3 Reducing resolution

720p → 480p typically doubles FPS with minimal accuracy loss for body-scale applications:

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### 6.4 Benchmark (approximate, 640x480)

| Model complexity | CPU FPS | GPU FPS (NVIDIA T4) |
|-----------------|---------|---------------------|
| 0 (lite) | 30 | 90 |
| 1 (full) | 18 | 60 |
| 2 (heavy) | 9 | 35 |

---

## Summary

In this tutorial you:
- Learned BlazePose's 33 landmark layout and model complexity options
- Implemented a reusable joint-angle calculator
- Built a bicep-curl rep counter with per-arm tracking
- Built a squat counter using bilateral knee angles
- Applied frame-skipping and threading patterns for real-time optimisation

Next: [Tutorial 05 – Object Detection](05_object_detection.md)
