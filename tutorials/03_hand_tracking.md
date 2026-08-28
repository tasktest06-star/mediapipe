# Tutorial 03: Hand Tracking

## Introduction

Hand tracking is one of MediaPipe's most popular solutions. The Hand Landmark model detects
the 21 keypoints of a hand skeleton from a single RGB image — no depth sensor required — at
30+ FPS on a modern CPU. Applications range from sign-language recognition and virtual
musical instruments to surgical training simulators and touchless kiosk interfaces.

This tutorial builds four progressively complex applications:
1. Single and multi-hand detection with landmark overlay
2. A complete rock/paper/scissors gesture recogniser
3. A finger-counting application
4. A gesture-based mouse control sketch

---

## 1. Hand Landmark Model: 21 Keypoints

The model returns 21 3D landmarks per detected hand. Landmark indices follow this layout:

```
                8   12  16  20
                |   |   |   |
                7   11  15  19
            4   6   10  14  18
            |   5   9   13  17
            3   |   |   |   |
            |   +-----------+
            2       |
            |       0 (WRIST)
            1
```

Named landmark groups (from MediaPipe's `HandLandmark` enum):

| Index | Name | Region |
|-------|------|--------|
| 0 | WRIST | Wrist |
| 1–4 | THUMB_CMC → THUMB_TIP | Thumb |
| 5–8 | INDEX_FINGER_MCP → INDEX_FINGER_TIP | Index |
| 9–12 | MIDDLE_FINGER_MCP → MIDDLE_FINGER_TIP | Middle |
| 13–16 | RING_FINGER_MCP → RING_FINGER_TIP | Ring |
| 17–20 | PINKY_MCP → PINKY_TIP | Pinky |

Landmarks are normalised: x, y in [0,1] relative to image dimensions; z is depth relative
to the wrist (negative = closer to camera).

---

## 2. Single and Multi-Hand Detection

```python
"""
Tutorial 03 – Hand Landmark Overlay
Detects up to 2 hands and draws skeleton.
"""

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def main():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,       # 0=lite, 1=full
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    # Draw connections and landmark dots
                    mp_draw.draw_landmarks(
                        frame, hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

                # Show hand labels (Left/Right)
                if results.multi_handedness:
                    for i, handedness in enumerate(results.multi_handedness):
                        label = handedness.classification[0].label
                        score = handedness.classification[0].score
                        lms = results.multi_hand_landmarks[i].landmark
                        h, w, _ = frame.shape
                        cx = int(lms[0].x * w)
                        cy = int(lms[0].y * h)
                        cv2.putText(frame, f"{label} ({score:.0%})",
                                    (cx - 30, cy + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 255, 0), 2)

            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

---

## 3. Left/Right Hand Classification

MediaPipe reports handedness separately from landmarks. Note that because the image is
mirrored (common in selfie mode), left and right are **swapped** relative to the viewer.

```python
def get_handedness(results):
    """
    Returns list of (hand_landmarks, label, score) tuples.
    label is 'Left' or 'Right' from MediaPipe's perspective
    (opposite of viewer's perspective in mirrored mode).
    """
    output = []
    if not results.multi_handedness:
        return output
    for lms, hand in zip(results.multi_hand_landmarks,
                         results.multi_handedness):
        label = hand.classification[0].label
        score = hand.classification[0].score
        output.append((lms, label, score))
    return output
```

To correct for mirror flip, swap 'Left' and 'Right' after the call if you flipped the frame
with `cv2.flip(frame, 1)`.

---

## 4. Finger-Counting Application

A finger is considered "up" when its tip landmark has a lower y-coordinate than its MCP
(knuckle) landmark. The thumb uses x-coordinates instead (lateral movement).

```python
"""
Tutorial 03 – Finger Counter
Counts the number of raised fingers on each detected hand.
"""

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Tip and MCP landmark indices per finger
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_MCPS = [2, 5, 9, 13, 17]  # for thumb use x; others use y

def count_fingers(hand_lms, label, w, h):
    """
    Returns the number of raised fingers (0-5).
    label: 'Left' or 'Right' (MediaPipe convention, before mirror correction)
    """
    lm = hand_lms.landmark
    fingers_up = 0

    # Thumb: compare x coordinates
    # For Right hand (viewer's left after mirror), tip is to the left of MCP when up
    if label == "Right":
        if lm[FINGER_TIPS[0]].x < lm[FINGER_MCPS[0]].x:
            fingers_up += 1
    else:
        if lm[FINGER_TIPS[0]].x > lm[FINGER_MCPS[0]].x:
            fingers_up += 1

    # Other 4 fingers: tip y < mcp y means finger is up
    for tip, mcp in zip(FINGER_TIPS[1:], FINGER_MCPS[1:]):
        if lm[tip].y < lm[mcp].y:
            fingers_up += 1

    return fingers_up

def main():
    cap = cv2.VideoCapture(0)
    COUNT_COLORS = [
        (0,0,255),(0,128,255),(0,255,128),
        (0,255,0),(128,255,0),(255,255,0)
    ]

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            total = 0
            if results.multi_hand_landmarks and results.multi_handedness:
                for lms, handedness in zip(results.multi_hand_landmarks,
                                           results.multi_handedness):
                    label = handedness.classification[0].label
                    count = count_fingers(lms, label, w, h)
                    total += count

                    mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

                    cx = int(lms.landmark[0].x * w)
                    cy = int(lms.landmark[0].y * h)
                    cv2.putText(frame, str(count), (cx - 15, cy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                COUNT_COLORS[count], 3)

            # Big total display
            cv2.rectangle(frame, (0, 0), (120, 80), (30, 30, 30), -1)
            cv2.putText(frame, str(total), (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5,
                        COUNT_COLORS[min(total, 5)], 4)

            cv2.imshow("Finger Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

---

## 5. Rock / Paper / Scissors Gesture Recogniser

This recogniser maps finger-count patterns to RPS gestures:

| Gesture | Fingers up |
|---------|-----------|
| Rock | 0 |
| Paper | 5 |
| Scissors | 2 (index + middle) |

```python
"""
Tutorial 03 – Rock / Paper / Scissors Recogniser
Full working gesture classifier with auto-play logic.
"""

import cv2
import mediapipe as mp
import random
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_MCPS = [2, 5,  9, 13, 17]

def get_fingers_state(lms, label):
    """Returns bitmask: index [thumb, index, middle, ring, pinky]"""
    state = []
    lm = lms.landmark
    # Thumb
    if label == "Right":
        state.append(1 if lm[4].x < lm[2].x else 0)
    else:
        state.append(1 if lm[4].x > lm[2].x else 0)
    # Other fingers
    for tip, mcp in zip(FINGER_TIPS[1:], FINGER_MCPS[1:]):
        state.append(1 if lm[tip].y < lm[mcp].y else 0)
    return state

def classify_gesture(state):
    """Map finger state to RPS gesture string."""
    total = sum(state)
    index_up  = state[1]
    middle_up = state[2]
    ring_up   = state[3]
    pinky_up  = state[4]

    if total == 0:
        return "Rock"
    if total == 5:
        return "Paper"
    if index_up and middle_up and not ring_up and not pinky_up:
        return "Scissors"
    return "Unknown"

def determine_winner(player, computer):
    if player == computer:
        return "Draw"
    wins = {("Rock","Scissors"), ("Paper","Rock"), ("Scissors","Paper")}
    return "You Win!" if (player, computer) in wins else "CPU Wins!"

CHOICES = ["Rock", "Paper", "Scissors"]

def main():
    cap = cv2.VideoCapture(0)
    cpu_choice = random.choice(CHOICES)
    result_text = ""
    last_play_time = 0
    play_interval = 3.0   # seconds between auto-plays

    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            player_gesture = "No hand"
            if results.multi_hand_landmarks and results.multi_handedness:
                lms = results.multi_hand_landmarks[0]
                label = results.multi_handedness[0].classification[0].label
                state = get_fingers_state(lms, label)
                player_gesture = classify_gesture(state)
                mp_draw.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

            # Auto-play every N seconds
            now = time.time()
            if now - last_play_time > play_interval:
                if player_gesture not in ("No hand", "Unknown"):
                    cpu_choice = random.choice(CHOICES)
                    result_text = determine_winner(player_gesture, cpu_choice)
                    last_play_time = now

            # HUD
            cv2.rectangle(frame, (0, h-120), (w, h), (20,20,20), -1)
            cv2.putText(frame, f"You: {player_gesture}", (10, h-80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,128), 2)
            cv2.putText(frame, f"CPU: {cpu_choice}", (10, h-45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,128,0), 2)
            cv2.putText(frame, result_text, (w//2 - 80, h-45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

            cv2.imshow("Rock Paper Scissors", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

---

## 6. Gesture-Based Mouse Control

This sketch maps hand position and pinch gesture to mouse movement and clicking. It uses the
`pyautogui` library for OS-level mouse control.

```python
"""
Tutorial 03 – Gesture Mouse Control Sketch
Maps hand position to cursor; pinch index+thumb to click.
Requires: pip install pyautogui
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np

mp_hands = mp.solutions.hands

# Smooth the cursor to avoid jitter
SMOOTH = 0.2
prev_x, prev_y = 0, 0
CLICK_THRESHOLD = 0.04   # normalised distance for pinch detection

def get_landmark_xy(lms, idx, w, h):
    lm = lms.landmark[idx]
    return lm.x * w, lm.y * h

def main():
    global prev_x, prev_y
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    screen_w, screen_h = pyautogui.size()
    pyautogui.FAILSAFE = False

    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:

        clicking = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                lms = results.multi_hand_landmarks[0]

                # Index fingertip controls cursor
                ix, iy = get_landmark_xy(lms, 8, w, h)
                tx, ty = get_landmark_xy(lms, 4, w, h)

                # Map to screen coordinates (with margin to avoid edges)
                margin = 0.1
                norm_x = np.clip((ix / w - margin) / (1 - 2*margin), 0, 1)
                norm_y = np.clip((iy / h - margin) / (1 - 2*margin), 0, 1)
                target_x = int(norm_x * screen_w)
                target_y = int(norm_y * screen_h)

                # Smooth movement
                cur_x = int(prev_x + SMOOTH * (target_x - prev_x))
                cur_y = int(prev_y + SMOOTH * (target_y - prev_y))
                prev_x, prev_y = cur_x, cur_y
                pyautogui.moveTo(cur_x, cur_y)

                # Pinch to click
                pinch_dist = np.hypot(ix - tx, iy - ty) / w
                if pinch_dist < CLICK_THRESHOLD and not clicking:
                    pyautogui.click()
                    clicking = True
                elif pinch_dist >= CLICK_THRESHOLD:
                    clicking = False

                mp_hands.solutions.drawing_utils.draw_landmarks(
                    frame, lms, mp_hands.HAND_CONNECTIONS
                )

                # Visual feedback
                color = (0, 0, 255) if clicking else (0, 255, 0)
                cv2.circle(frame, (int(ix), int(iy)), 12, color, 3)

            cv2.imshow("Gesture Mouse", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

---

## 7. Performance Tips

### model_complexity

- `model_complexity=0` (lite) — faster, slightly less accurate. Use for real-time apps on
  CPU.
- `model_complexity=1` (full) — more accurate, recommended for static image analysis.

### max_num_hands

Each additional hand adds roughly 40% to inference time. Limit to your application needs.

### Hand detection vs tracking

MediaPipe runs the full detector only when a hand is first seen (or tracking is lost).
Subsequent frames use a lighter tracker. This means the first frame is slower — warm up the
model with a dummy frame if needed:

```python
# Warm-up: run one blank frame before the main loop
blank = np.zeros((480, 640, 3), dtype=np.uint8)
hands.process(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB))
```

### FPS benchmark (approximate, 640x480, Intel i7-12700)

| Hands | Complexity | FPS (CPU) |
|-------|-----------|-----------|
| 1 | 0 (lite) | 65 |
| 1 | 1 (full) | 38 |
| 2 | 0 (lite) | 45 |
| 2 | 1 (full) | 22 |

---

## Summary

In this tutorial you:
- Detected hands and drew the 21-keypoint skeleton
- Classified left/right handedness
- Built a finger counter using tip-vs-MCP y-coordinate comparisons
- Created a complete RPS game with automatic CPU opponent
- Sketched a gesture-driven mouse controller using index-finger tracking and
  thumb-index pinch detection

Next: [Tutorial 04 – Pose Estimation](04_pose_estimation.md)
