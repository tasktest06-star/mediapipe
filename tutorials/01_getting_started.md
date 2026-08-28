# Tutorial 01: Getting Started with MediaPipe

## Introduction

MediaPipe is one of the most powerful and versatile on-device ML frameworks available today.
Whether you are building a fitness app that counts reps, a video-conferencing tool that blurs
backgrounds, or a sign-language interpreter, MediaPipe provides battle-tested, hardware-
accelerated perception pipelines that run in real time on everything from a Raspberry Pi to a
high-end workstation.

This tutorial walks you from zero — a fresh Python environment — to running a fully working
face-detection demo. Along the way you will understand *why* MediaPipe works the way it does,
not just *how* to call its API, so that later tutorials feel natural rather than magical.

---

## 1. Installation

### 1.1 Via pip (recommended)

The simplest path for most developers:

```bash
pip install mediapipe
```

If you need OpenCV for camera access and image display (almost always yes):

```bash
pip install mediapipe opencv-python numpy
```

For Jupyter notebooks add:

```bash
pip install mediapipe opencv-python numpy ipywidgets
```

### 1.2 Via conda

MediaPipe is not in the default Anaconda channel, but you can use pip inside a conda
environment:

```bash
conda create -n mediapipe python=3.10
conda activate mediapipe
pip install mediapipe opencv-python numpy
```

### 1.3 From source (advanced)

Building from source is only necessary if you need custom calculators, a platform not covered
by the pre-built wheels (e.g., ARMv7 Linux), or the very latest unreleased code.

```bash
# Prerequisites: Bazel 6.x, Python 3.10, OpenCV from source
git clone https://github.com/google/mediapipe.git
cd mediapipe

# Install Python dependencies
pip install -r requirements.txt

# Build the Python package wheel
python setup.py gen_protos && pip install -e .
```

Expect the first build to take 20-40 minutes on a modern machine.

### 1.4 Verifying your installation

```python
import mediapipe as mp
import cv2
import numpy as np

print("MediaPipe version:", mp.__version__)
print("OpenCV version   :", cv2.__version__)
print("NumPy version    :", np.__version__)
```

You should see output similar to:
```
MediaPipe version: 0.10.9
OpenCV version   : 4.9.0
NumPy version    : 1.26.4
```

---

## 2. Core Concepts

Before writing a single line of application code, it is worth spending a few minutes on the
mental model that underlies every MediaPipe solution.

### 2.1 Graphs

A MediaPipe **graph** is a directed network of processing nodes called *calculators*, described
in a Protocol Buffer text format (`.pbtxt`). Each edge in the graph is a **stream** — an
ordered sequence of typed, timestamped values called **packets**.

You can think of a graph as an assembly line:
- Raw frames enter the line at the *source* end.
- Each station (calculator) does one focused job: resize, normalise, run inference, decode
  detections, draw overlays.
- Finished results (landmarks, bounding boxes, segmentation masks) exit at the *sink* end.

The power of this model is composability: you can swap an inference calculator for a different
model, insert a new pre-processing step, or branch the pipeline for parallel processing —
all without rewriting downstream code.

### 2.2 Calculators

A **calculator** is a C++ class (Python wrappers exist) that declares:
- `GetContract()` — the types of its input/output streams and side packets.
- `Open()` — one-time initialisation (load model, allocate buffers).
- `Process()` — per-packet logic (called once per timestamp).
- `Close()` — cleanup.

The MediaPipe runtime schedules calculators automatically, running them on the same or
different threads based on the graph topology and resource constraints.

### 2.3 Packets

A **packet** bundles a value together with a timestamp:

```
Packet = (timestamp_microseconds, typed_value)
```

Common packet types: `ImageFrame`, `NormalizedLandmarkList`, `DetectionList`,
`ClassificationList`, `float`, `bool`.

The timestamp is crucial: it is how the framework synchronises packets from different streams
(e.g., matching the camera frame with its corresponding detections).

### 2.4 Streams and Side Packets

- **Input stream** — time-varying data (each frame is one packet).
- **Output stream** — results produced for each input.
- **Side packet** — a single value provided at graph startup, not time-varying. Used for
  configuration: model path, confidence threshold, maximum number of detections.

---

## 3. Solutions API vs Tasks API

MediaPipe ships two Python-level abstractions.

### 3.1 Solutions API (legacy, pre-0.10)

The Solutions API wraps individual sub-graphs (FaceMesh, Hands, Pose, Holistic …) as Python
context managers:

```python
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    min_detection_confidence=0.5
) as face_mesh:
    result = face_mesh.process(rgb_image)
```

The Solutions API is mature, widely documented, and still works in MediaPipe 0.10.x, but
Google has marked it as legacy and is not adding new models to it.

### 3.2 Tasks API (current, 0.10+)

The Tasks API provides a unified interface across all platforms (Python, Android, iOS, Web)
and supports the new model-bundle format (`.task` files):

```python
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Load the model
base_options = mp_python.BaseOptions(
    model_asset_path='face_detector.task'
)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# Run inference
mp_image = mp.Image.create_from_file('photo.jpg')
result = detector.detect(mp_image)
```

Tasks API highlights:
- **Consistent across languages** — the same concepts map to Android/iOS/Web.
- **Live-stream mode** — results are delivered via callback for minimal latency.
- **Video mode** — processes whole video files efficiently.
- **Task bundles** — model + metadata in one `.task` file, downloaded from
  https://storage.googleapis.com/mediapipe-models/

For **new projects** use the Tasks API. The tutorials in this series show both APIs where
relevant.

---

## 4. Hello-World: Face Detection

The following complete program opens your webcam, detects faces in every frame, draws a
bounding box around each face, and displays the result. Press **Q** to quit.

```python
"""
Tutorial 01 – Hello-World Face Detection
Uses the MediaPipe Solutions API (legacy) for simplicity.
"""

import cv2
import mediapipe as mp

# --- Setup ---
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def draw_detections(frame, detections):
    """Draw bounding boxes and confidence scores on frame (in-place)."""
    h, w, _ = frame.shape
    for detection in detections:
        bbox = detection.location_data.relative_bounding_box
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        score = detection.score[0]
        label = f"{score:.0%}"

        cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check camera index or permissions.")

    with mp_face_detection.FaceDetection(
        model_selection=0,          # 0 = short-range (< 2 m), 1 = full-range (< 5 m)
        min_detection_confidence=0.5
    ) as face_detector:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe expects RGB; OpenCV delivers BGR
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False      # performance: avoid copy

            results = face_detector.process(rgb)

            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if results.detections:
                draw_detections(frame, results.detections)

            # FPS overlay
            cv2.putText(frame, "Press Q to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("MediaPipe Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

### What each line does

| Line(s) | Purpose |
|---------|---------|
| `mp_face_detection.FaceDetection(...)` | Instantiates the detector, loads the BlazeFace model into memory. |
| `model_selection=0` | Selects the short-range BlazeFace variant (faster, better for close-up selfie use cases). |
| `cv2.cvtColor(...BGR2RGB)` | Converts from OpenCV's BGR colour order to RGB which MediaPipe expects. |
| `rgb.flags.writeable = False` | Avoids an internal NumPy copy, saving ~1 ms per frame. |
| `face_detector.process(rgb)` | Runs the full graph: preprocessing → inference → decoding. |
| `results.detections` | A list of `Detection` proto messages, or `None` if no face was found. |

---

## 5. Understanding the Output

The `results.detections` list contains `Detection` objects, each with:

```python
detection.score            # list[float]: confidence (one per class)
detection.label_id         # list[int]:   class id
detection.location_data    # LocationData proto
  .relative_bounding_box   # BoundingBox in [0,1] coordinates
    .xmin, .ymin           # top-left corner
    .width, .height        # size
  .relative_keypoints      # list of 6 facial keypoints (right eye, left eye,
                           #   nose tip, mouth centre, right ear, left ear)
```

All coordinates are *relative* (0.0 – 1.0 fraction of image width/height). Multiply by image
dimensions to get pixel coordinates.

---

## 6. Troubleshooting

### 6.1 Camera permissions

**Linux:** Add your user to the `video` group:
```bash
sudo usermod -aG video $USER
# Log out and back in, then:
ls -la /dev/video*   # should show rw permissions
```

**macOS:** System Settings → Privacy & Security → Camera → enable for Terminal / your IDE.

**Windows:** Settings → Privacy → Camera → allow apps to access camera.

### 6.2 Wrong webcam index

If `VideoCapture(0)` shows a black screen or wrong camera, try indices 1, 2, 3 …
On Linux you can list available devices:
```bash
v4l2-ctl --list-devices
```

### 6.3 Model download failures

MediaPipe downloads model files on first use. If you are behind a corporate proxy:
```bash
export HTTPS_PROXY=http://proxy.example.com:8080
python your_script.py
```

Alternatively, download the model manually and pass the local path via `model_asset_path`.

### 6.4 GPU acceleration not working

MediaPipe uses the GPU automatically when available on Android/iOS. On desktop Python, GPU
support depends on your platform:
- **Linux + NVIDIA:** Install CUDA 11.x + cuDNN 8, then `pip install mediapipe-gpu` (if
  available for your version).
- **macOS (Apple Silicon):** GPU is used automatically via Metal since mediapipe 0.10.5.
- **Windows:** CPU inference is the default; GPU support is experimental.

Check effective backend:
```python
# No direct API — monitor with nvidia-smi or Activity Monitor to confirm GPU usage
```

### 6.5 ImportError: cannot import name '...'

Usually a version mismatch. Check:
```bash
pip show mediapipe   # installed version
pip install --upgrade mediapipe
```

### 6.6 Slow frame rate

- Ensure you are passing `rgb.flags.writeable = False` before `process()`.
- Reduce input resolution: `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)`.
- Use `model_selection=0` (short-range, faster) when faces are close to the camera.
- On multi-core machines MediaPipe parallelises automatically; ensure no other process
  monopolises CPU cores.

---

## 7. Next Steps

You now have a working face-detection pipeline. In the next tutorial you will go deeper:
detecting the 478 facial landmarks of Face Mesh, estimating face distance, and tracking
multiple faces simultaneously.

Proceed to [Tutorial 02: Face Detection & Mesh](02_face_detection.md).
