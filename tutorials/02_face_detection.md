# Tutorial 02: Face Detection and Face Mesh

## Introduction

Human faces are the most information-dense region of the human body — they convey identity,
emotion, gaze direction, age, and health cues simultaneously. MediaPipe provides two
complementary face models:

1. **Face Detection (BlazeFace)** — an ultra-fast bounding-box detector that locates faces
   and 6 keypoints in under 1 ms on a mobile GPU.
2. **Face Mesh** — a heavier model that fits 478 3D landmarks to the detected face, enabling
   applications from AR makeup to clinical face analysis.

In this tutorial you will build:
- A real-time webcam face-detection loop with FPS counter
- A face-mesh overlay showing all 478 landmarks
- A face-distance estimator using inter-pupillary distance (IPD)
- A multi-face tracker that assigns stable IDs
- A CSV exporter for offline landmark analysis

---

## 1. BlazeFace: The Face Detection Model

BlazeFace is a MobileNet-based, SSD-style detector optimised for mobile hardware. MediaPipe
ships two variants:

| Parameter | `model_selection=0` | `model_selection=1` |
|-----------|--------------------|--------------------|
| Name | Short-range | Full-range |
| Best range | < 2 m | < 5 m |
| Speed | ~1 ms (GPU) | ~2 ms (GPU) |
| Use case | Selfie, video-call | Surveillance, group shots |

Each detection includes:
- `relative_bounding_box` — (xmin, ymin, width, height) in [0,1] coordinates
- `score` — confidence value in [0,1]
- `relative_keypoints` — 6 keypoints: right eye, left eye, nose tip, mouth centre,
  right ear tragion, left ear tragion

### Minimal face detection

```python
import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = fd.process(rgb)

        if results.detections:
            for det in results.detections:
                mp_draw.draw_detection(frame, det)

        cv2.imshow("BlazeFace", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

---

## 2. Face Mesh: 478 Landmarks

Face Mesh fits a 3D mesh to the detected face and returns 478 landmarks in normalised
coordinates (x, y in [0,1], z relative to camera).

The 478 points cover:
- Facial outline (jaw, cheekbones, temples)
- Eyes: inner/outer corners, upper/lower lids, iris (5 iris landmarks per eye)
- Eyebrows: inner/outer ends, arch
- Nose: bridge, tip, nostrils
- Lips: outer and inner contours, commissures
- Forehead

### Full Face Mesh webcam demo

```python
"""
Tutorial 02 - Face Mesh Webcam Demo
Draws all 478 landmarks and the tessellation mesh.
"""

import cv2
import mediapipe as mp
import time

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Drawing specs
TESSELATION_SPEC = mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1)
CONTOUR_SPEC = mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2)
LANDMARK_SPEC = mp_drawing.DrawingSpec(
    color=(0, 128, 255), thickness=1, circle_radius=1
)

def compute_fps(prev_time):
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time + 1e-9)
    return fps, curr_time

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    prev_time = time.time()

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=4,
        refine_landmarks=True,    # enables iris landmarks (468-477)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_lms in results.multi_face_landmarks:
                    # Draw tessellation
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_lms,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=TESSELATION_SPEC,
                    )
                    # Draw contours (eyes, lips, face oval)
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_lms,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=CONTOUR_SPEC,
                    )
                    # Draw irises (requires refine_landmarks=True)
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_lms,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                            .get_default_face_mesh_iris_connections_style(),
                    )

            fps, prev_time = compute_fps(prev_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Face Mesh", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

---

## 3. Face Distance Estimation via Inter-Pupillary Distance

The inter-pupillary distance (IPD) in pixels shrinks as a face moves away from the camera.
Given a known physical IPD (average adult: ~63 mm), you can estimate depth using the
thin-lens formula:

```
distance_mm = (focal_length_px * real_IPD_mm) / ipd_pixels
```

where `focal_length_px` is calibrated (or approximated from `frame_width * 0.7`).

```python
"""
Tutorial 02 - Face Distance Estimator
Estimates distance to face using inter-pupillary distance.
"""

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# Landmark indices for left and right iris centres
# (requires refine_landmarks=True)
LEFT_IRIS_IDX = 468     # centre of left iris
RIGHT_IRIS_IDX = 473    # centre of right iris

REAL_IPD_MM = 63.0      # average adult IPD in mm

def get_iris_coords(landmarks, idx, w, h):
    lm = landmarks.landmark[idx]
    return int(lm.x * w), int(lm.y * h)

def estimate_distance(frame, face_landmarks, focal_length_px):
    h, w, _ = frame.shape
    lx, ly = get_iris_coords(face_landmarks, LEFT_IRIS_IDX, w, h)
    rx, ry = get_iris_coords(face_landmarks, RIGHT_IRIS_IDX, w, h)
    ipd_px = np.sqrt((lx - rx)**2 + (ly - ry)**2)
    if ipd_px < 1:
        return None
    distance = (focal_length_px * REAL_IPD_MM) / ipd_px
    return distance

def main():
    cap = cv2.VideoCapture(0)
    ret, sample = cap.read()
    h, w, _ = sample.shape
    focal_length_px = w * 0.7   # rough approximation; calibrate for accuracy

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0]
                dist = estimate_distance(frame, lms, focal_length_px)
                if dist:
                    label = f"Distance: {dist/10:.1f} cm"
                    cv2.putText(frame, label, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 200, 255), 2)

            cv2.imshow("Distance Estimator", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

---

## 4. Multi-Face Tracking with Stable IDs

MediaPipe Face Mesh tracks multiple faces but does not assign persistent IDs across frames.
The following implementation uses IoU (Intersection-over-Union) matching to maintain stable
face IDs across frames.

```python
"""
Tutorial 02 - Multi-Face Tracker with IDs
Assigns stable IDs to faces across frames using IoU matching.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import OrderedDict

mp_face_detection = mp.solutions.face_detection

class FaceTracker:
    def __init__(self, iou_threshold=0.3, max_lost=30):
        self.next_id = 0
        self.faces = OrderedDict()   # id -> (bbox, lost_count)
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

    @staticmethod
    def bbox_to_rect(bbox, w, h):
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)
        return x1, y1, x2, y2

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / (union + 1e-6)

    def update(self, detections, w, h):
        rects = [self.bbox_to_rect(d.location_data.relative_bounding_box, w, h)
                 for d in detections]

        if not self.faces:
            for r in rects:
                self.faces[self.next_id] = (r, 0)
                self.next_id += 1
            return dict(self.faces)

        # Match by max IoU
        updated = {}
        used_ids = set()
        for r in rects:
            best_id, best_iou = -1, self.iou_threshold
            for fid, (fr, _) in self.faces.items():
                if fid in used_ids:
                    continue
                score = self.iou(r, fr)
                if score > best_iou:
                    best_iou = score
                    best_id = fid
            if best_id == -1:
                best_id = self.next_id
                self.next_id += 1
            updated[best_id] = (r, 0)
            used_ids.add(best_id)

        # Increment lost count for unmatched
        for fid, (fr, lost) in self.faces.items():
            if fid not in updated:
                if lost < self.max_lost:
                    updated[fid] = (fr, lost + 1)

        self.faces = OrderedDict(updated)
        return dict(self.faces)

def main():
    cap = cv2.VideoCapture(0)
    tracker = FaceTracker()
    COLORS = [(0,255,0),(255,128,0),(0,128,255),(255,0,128),(128,0,255)]

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as fd:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = fd.process(rgb)
            detections = results.detections or []
            tracked = tracker.update(detections, w, h)

            for fid, (rect, lost) in tracked.items():
                if lost > 0:
                    continue
                x1, y1, x2, y2 = rect
                color = COLORS[fid % len(COLORS)]
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, f"ID {fid}", (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Multi-Face Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

---

## 5. Saving Landmark Data to CSV

For offline analysis, biomechanics research, or training a downstream model, you can export
all 478 landmark coordinates to a CSV file.

```python
"""
Tutorial 02 - Landmark CSV Exporter
Saves face mesh landmarks to a CSV for each frame.
"""

import cv2
import mediapipe as mp
import csv
import time

mp_face_mesh = mp.solutions.face_mesh

NUM_LANDMARKS = 478
HEADER = ['timestamp'] + [
    f'lm{i}_{axis}' for i in range(NUM_LANDMARKS) for axis in ('x','y','z')
]

def export_landmarks(filename='face_landmarks.csv', duration_sec=30):
    cap = cv2.VideoCapture(0)
    start = time.time()

    with (mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
          ) as face_mesh,
          open(filename, 'w', newline='') as f):

        writer = csv.writer(f)
        writer.writerow(HEADER)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            elapsed = time.time() - start
            if elapsed > duration_sec:
                print(f"Captured {duration_sec}s of data -> {filename}")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                row = [f"{elapsed:.4f}"]
                for lm in lms:
                    row += [f"{lm.x:.6f}", f"{lm.y:.6f}", f"{lm.z:.6f}"]
                writer.writerow(row)

            cv2.imshow("Recording...", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    export_landmarks()
```

---

## 6. Performance Tips

### Reduce input resolution

The face mesh model scales with image size. For real-time apps, 640x480 is usually sufficient:

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### Limit max_num_faces

Each additional face adds ~30% inference time. Set `max_num_faces` to the minimum needed.

### Use static_image_mode wisely

- `static_image_mode=False` (default) — runs detection only when tracking is lost. Faster for
  video streams.
- `static_image_mode=True` — runs detection on every image. Use only for single static images.

### Disable iris landmarks when not needed

`refine_landmarks=True` adds 10 iris keypoints per face and ~5 ms overhead. Disable if you
do not need gaze or blink detection.

### Threading pattern

Run frame capture in a background thread to decouple I/O latency from inference latency:

```python
import threading, queue

frame_q = queue.Queue(maxsize=2)

def capture_worker(cap, q):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not q.full():
            q.put(frame)

cap = cv2.VideoCapture(0)
t = threading.Thread(target=capture_worker, args=(cap, frame_q), daemon=True)
t.start()

while True:
    if frame_q.empty():
        continue
    frame = frame_q.get()
    # ... inference ...
```

### Benchmark numbers (approximate, 720p, MacBook Pro M2)

| Solution | FPS (CPU) | FPS (GPU/Metal) |
|----------|-----------|-----------------|
| FaceDetection short-range | 120+ | 200+ |
| FaceMesh 1 face | 55 | 95 |
| FaceMesh 4 faces | 22 | 65 |

---

## Summary

In this tutorial you:
- Used BlazeFace to detect faces with 6 facial keypoints
- Applied Face Mesh to obtain 478 3D landmarks
- Built a face-distance estimator using iris IPD
- Implemented a simple IoU-based multi-face ID tracker
- Exported landmark streams to CSV for offline analysis

Next: [Tutorial 03 – Hand Tracking](03_hand_tracking.md)
