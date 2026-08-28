# MediaPipe Tutorials

A comprehensive tutorial series covering MediaPipe's most powerful computer-vision solutions,
from installation through production-grade applications.

---

## What is MediaPipe?

MediaPipe is Google's open-source, cross-platform framework for building real-time perception
pipelines. Originally developed to power Google's own products (Google Meet background
segmentation, Pixel camera features, YouTube AR effects), it was open-sourced in 2019 and has
since become the de-facto toolkit for on-device ML inference in mobile, web, desktop, and
embedded environments.

At its core, MediaPipe wraps pre-trained, highly optimised ML models inside a **graph-based
dataflow engine**. Each graph node is a *calculator* that receives typed *packets* from input
*streams*, transforms them, and emits packets on output streams. The framework handles
threading, back-pressure, GPU/CPU routing, and model loading automatically, so application
developers can focus on what the data means rather than how it flows.

---

## Pipeline Architecture Overview

```
                     ┌─────────────────────────────────────────────────────┐
                     │              MediaPipe Graph                         │
                     │                                                       │
  ┌──────────┐       │  ┌───────────┐   ┌───────────┐   ┌───────────┐     │   ┌──────────────┐
  │  Input   │       │  │ Image     │   │  Model    │   │  Post-    │     │   │   Output     │
  │  Source  ├──────►│  │ Pre-proc  ├──►│ Inference ├──►│  process  ├─────┼──►│  (landmarks  │
  │ (camera/ │       │  │ Calculator│   │ Calculator│   │ Calculator│     │   │   boxes,     │
  │  video/  │       │  └───────────┘   └───────────┘   └───────────┘     │   │   masks …)   │
  │  image)  │       │        ▲               ▲               ▲            │   └──────────────┘
  └──────────┘       │        │               │               │            │
                     │  ┌─────┴───────────────┴───────────────┴──────┐    │
                     │  │            Side-Packet Inputs               │    │
                     │  │  (model files, config options, thresholds)  │    │
                     │  └─────────────────────────────────────────────┘    │
                     └─────────────────────────────────────────────────────┘
```

Key concepts:
- **Graph (.pbtxt)** – a directed acyclic (or with feedback loops) description of calculators
  and the streams connecting them.
- **Calculator** – a C++ (or Python-wrapped) class that processes one or more input stream
  packets and produces output packets. Stateless or stateful.
- **Packet** – a timestamped, strongly-typed datum travelling on a stream.
- **Stream** – a time-ordered sequence of packets connecting two calculators.
- **Side packet** – a one-time value injected at graph startup (e.g., model path, threshold).

---

## Tutorial List

| # | Title | Description |
|---|-------|-------------|
| 01 | [Getting Started](tutorials/01_getting_started.md) | Install MediaPipe, understand the graph model, run a Hello-World face detection example, and learn the difference between the Solutions API and the Tasks API. Includes a detailed troubleshooting section covering camera permissions, model downloads, and GPU setup. |
| 02 | [Face Detection & Mesh](tutorials/02_face_detection.md) | Explore the BlazeFace detector and Face Mesh model (478 landmarks). Build a real-time webcam loop, estimate face distance from inter-pupillary distance, track multiple faces with IDs, and export landmark data to CSV for downstream analysis. |
| 03 | [Hand Tracking](tutorials/03_hand_tracking.md) | Work with the 21-keypoint Hand Landmark model to build a complete rock/paper/scissors gesture recogniser, a finger-counting application, left/right hand classification logic, and a gesture-driven mouse-control sketch that can be adapted for accessibility tools. |
| 04 | [Pose Estimation](tutorials/04_pose_estimation.md) | Use BlazePose's 33 body landmarks to build a bicep-curl rep counter and a squat counter complete with angle-threshold feedback. Learn joint-angle calculation utilities, full-body vs upper-body model selection, and real-time optimisation with frame-skipping and threading. |
| 05 | [Object Detection](tutorials/05_object_detection.md) | Compare EfficientDet-Lite0/1/2 for accuracy/speed trade-offs, build a real-time bounding-box detector, tune score thresholds and NMS parameters, implement a centroid tracker for multi-object tracking, count objects by class, and benchmark FPS across CPU/GPU/EdgeTPU hardware. |
| 06 | [Image Segmentation](tutorials/06_image_segmentation.md) | Apply selfie segmentation and multi-class segmentation models. Build real-time background replacement with OpenCV, create a virtual green-screen pipeline for video streams, and fuse segmentation masks with object-detection bounding boxes for richer scene understanding. |
| 07 | [Holistic Solution](tutorials/07_holistic_solution.md) | Run all sub-models simultaneously — face mesh, both hands, and full-body pose — with the Holistic API. Build a synchronised multi-landmark tracker, sketch an avatar controller, outline a sign-language recognition pipeline, and learn to budget CPU time across the concurrent sub-models. |

---

## Prerequisites

- Python 3.8 – 3.12
- A webcam (for live demos) or sample video/image files
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- Basic familiarity with Python and NumPy array indexing

Optional (for GPU acceleration):
- CUDA 11.x + cuDNN 8.x (Linux / Windows)
- Or an Apple Silicon Mac (Metal backend)

---

## Installation

```bash
# Recommended: create a virtual environment first
python -m venv mp-env
source mp-env/bin/activate          # Windows: mp-env\Scripts\activate

# Core install
pip install mediapipe

# Full stack for all tutorials
pip install mediapipe opencv-python numpy pandas matplotlib
```

### Verify installation

```python
import mediapipe as mp
print(mp.__version__)   # e.g., 0.10.x
```

---

## How to Run Each Tutorial

Every `.md` tutorial contains complete, self-contained Python code blocks.
Copy the code into a `.py` file (or a Jupyter notebook) and run it directly.

```bash
# Example – Tutorial 02, face mesh webcam demo
python face_mesh_demo.py

# Press Q in the OpenCV window to quit
```

All tutorials follow the same structure:

```python
import cv2
import mediapipe as mp

# 1. Initialise the solution
# 2. Open a VideoCapture (0 = default webcam)
# 3. Loop: read frame -> process -> draw -> imshow
# 4. Release resources
```

---

## Repository Structure

```
mediapipe-tutorials/
├── README.md
└── tutorials/
    ├── 01_getting_started.md / .tex
    ├── 02_face_detection.md  / .tex
    ├── 03_hand_tracking.md   / .tex
    ├── 04_pose_estimation.md / .tex
    ├── 05_object_detection.md/ .tex
    ├── 06_image_segmentation.md / .tex
    └── 07_holistic_solution.md  / .tex
```

---

## License

Tutorial content is released under CC-BY 4.0. Code samples are MIT licensed.
