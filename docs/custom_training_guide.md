# Custom Training Guide: EfficientDet-Lite2 with 150 Classes

This guide walks you through every step required to train a MediaPipe
EfficientDet-Lite2 object-detection model on **150 custom classes from scratch**.
By the end you will have a `.tflite` file ready to drop into a MediaPipe
object-detection pipeline on Android, iOS, or desktop.

---

## Prerequisites

Before starting, verify the following:

- **Python 3.9 or 3.10** (TensorFlow 2.13 does not support Python 3.11+)
- **CUDA 11.8 + cuDNN 8.6** installed and visible to TensorFlow
- **NVIDIA GPU with at least 16 GB VRAM** (RTX 3090, A6000, or A100 recommended)
- **32 GB system RAM** (64 GB preferred for large tf.data pipelines)
- **500 GB free SSD space** for images, TFRecords, and checkpoints
- **Docker** (optional but simplifies dependency management)

Check CUDA availability:

```bash
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected output:
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## Step 1: Setting Up the Environment

### 1.1 Create a virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 1.2 Install all dependencies

```bash
pip install -r requirements/python_requirements.txt
```

This installs TensorFlow 2.13, MediaPipe Model Maker 0.2.1.4, Albumentations,
and all other pinned dependencies.  The install typically takes 5-10 minutes on
a fast connection.

### 1.3 Verify the install

```bash
python - <<'EOF'
import tensorflow as tf
from mediapipe_model_maker import object_detector
import albumentations as A
print("TensorFlow:", tf.__version__)
print("All imports OK")
EOF
```

Expected output:
```
TensorFlow: 2.13.0
All imports OK
```

### 1.4 (Optional) Docker setup

```dockerfile
FROM tensorflow/tensorflow:2.13.0-gpu
WORKDIR /workspace
COPY requirements/python_requirements.txt .
RUN pip install -r python_requirements.txt
```

```bash
docker build -t mediapipe-training .
docker run --gpus all -v /data:/data -v $(pwd):/workspace mediapipe-training bash
```

---

## Step 2: Organising Your 150-Class Dataset

Your raw data should be organised as follows before annotation:

```
/data/
  raw_images/
    class_001_bicycle/
      img_0001.jpg
      img_0002.jpg
      ...
    class_002_car/
      ...
    ...
    class_150_umbrella/
      ...
  annotations/        <- will be populated in Step 3
```

### 2.1 Minimum counts

Per the data requirements (`requirements/data_requirements.txt`):
- Minimum: 300 images per class
- Preferred: 500-1000 images per class
- For 150 classes: 75,000 to 150,000 total labelled images

### 2.2 Checking counts quickly

```python
import os
import collections

raw_dir = "/data/raw_images"
counts = {}
for cls_dir in os.listdir(raw_dir):
    full = os.path.join(raw_dir, cls_dir)
    if os.path.isdir(full):
        n = len([f for f in os.listdir(full) if f.lower().endswith((".jpg", ".png"))])
        counts[cls_dir] = n

under = {k: v for k, v in counts.items() if v < 300}
print(f"Classes under 300 images: {len(under)}")
for name, cnt in sorted(under.items()):
    print(f"  {name}: {cnt}")
```

### 2.3 150-class specific considerations

With 150 classes you will almost certainly encounter:

**Fine-grained classes** (e.g. 20 breeds of dog): ensure images show the
distinguishing features clearly. Use close-up shots as well as full-scene images.

**Class imbalance**: use `prepare_dataset.py --validate` to check the imbalance
ratio before training. Random under-sampling is applied automatically if the
ratio exceeds 3:1.

**Visually similar classes**: include "hard negative" images where a visually
similar object that is NOT the target class appears without annotation. This
teaches the model the subtle differences.

---

## Step 3: Annotation Guidelines and Tools

### 3.1 Recommended tool: LabelMe

LabelMe saves annotations in a JSON format that can be converted to COCO.

```bash
# Install (already in requirements)
labelme

# Or run as a web app
labelme --port 12345
```

### 3.2 COCO JSON annotation structure

Your final annotation file should follow this structure:

```json
{
  "images": [
    {"id": 1, "file_name": "img_0001.jpg", "width": 1280, "height": 720}
  ],
  "categories": [
    {"id": 1, "name": "bicycle"},
    {"id": 2, "name": "car"}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [120, 80, 200, 150],
      "area": 30000,
      "iscrowd": 0
    }
  ]
}
```

The `bbox` field is `[x_min, y_min, width, height]` (COCO convention).

### 3.3 Box quality guidelines

- **Tight boxes**: allow at most 5 px padding on each side.
- **Minimum size**: 32 x 32 px in the source image.
- **Occluded objects**: if more than 50% of the object is occluded, skip it or
  mark `iscrowd: 1`.
- **Truncated objects**: objects at the image edge should be annotated with a
  box that extends to the image boundary.

### 3.4 Quality checklist

Before finalising annotations, verify all 10 items in
`requirements/data_requirements.txt` [QUALITY CHECKLIST].

---

## Step 4: Converting Annotations to TFRecord

`prepare_dataset.py` accepts COCO JSON, Pascal VOC XML, and YOLO TXT formats.

### 4.1 From COCO JSON

```bash
python src/prepare_dataset.py \
    --input_format coco \
    --annotations  /data/annotations/instances_all.json \
    --images_dir   /data/raw_images \
    --output_dir   /data/tfrecords \
    --splits 0.70 0.15 0.15 \
    --shards 32
```

Expected console output:
```
Loading coco dataset...

--- Dataset Validation ---
Class imbalance ratio: 2.4:1 (within 3:1 limit) [OK]
Total images: 112500
Total classes: 150
--------------------------

Class                                    Images
--------------------------------------------------
car                                       950
bicycle                                   820
...

Split sizes: train=78750, val=16875, test=16875
  [train] 78750 records written to /data/tfrecords/train
  [val]   16875 records written to /data/tfrecords/val
  [test]  16875 records written to /data/tfrecords/test
Class map written to /data/tfrecords/class_map.json
Dataset preparation complete.
```

### 4.2 From Pascal VOC

```bash
python src/prepare_dataset.py \
    --input_format   voc \
    --annotations_dir /data/voc_annotations \
    --images_dir      /data/raw_images \
    --output_dir      /data/tfrecords
```

### 4.3 From YOLO

```bash
python src/prepare_dataset.py \
    --input_format yolo \
    --labels_dir   /data/yolo_labels \
    --names_file   /data/classes.txt \
    --images_dir   /data/raw_images \
    --output_dir   /data/tfrecords
```

### 4.4 Verifying the output

```python
import tensorflow as tf

ds = tf.data.TFRecordDataset(["/data/tfrecords/train/shard-00000-of-00032.tfrecord"])
for raw in ds.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw.numpy())
    print(example)
```

---

## Step 5: Configuring the Model Architecture

Edit `configs/training_config.yaml` to match your hardware and goals.

### 5.1 Choosing the right EfficientDet-Lite variant

| Model | Input | Params | COCO mAP | Latency (CPU) |
|-------|-------|--------|----------|---------------|
| Lite0 | 320x320 | 3.4M | 26.4% | 37 ms |
| Lite1 | 384x384 | 4.4M | 31.5% | 49 ms |
| Lite2 | 512x512 | 5.3M | 36.0% | 69 ms |
| Lite3 | 640x640 | 7.9M | 39.9% | 116 ms |

For 150 classes, **Lite2 is the recommended starting point**. The larger input
size (512x512) gives better detection of small objects without being too slow
for mobile inference.

### 5.2 Key hyperparameters for 150 classes

```yaml
training:
  epochs: 100
  batch_size: 16       # reduce to 8 if VRAM < 16 GB
  learning_rate: 0.08  # SGD with cosine decay
  lr_warmup_epochs: 5  # ramp up LR for first 5 epochs
  early_stopping:
    patience: 15       # stop if val_AP doesn't improve for 15 epochs
```

### 5.3 Mixed precision (strongly recommended)

Set `mixed_precision: true` in the config. This halves VRAM usage and speeds up
training by ~30% on Ampere/Hopper GPUs with no significant accuracy loss.

---

## Step 6: Launching Training

```bash
python src/train_model.py \
    --config     configs/training_config.yaml \
    --output_dir /models/efficientdet_lite2_150cls
```

### 6.1 Expected console output (first few epochs)

```
GPUs detected: ['/physical_device:GPU:0']
Mixed precision (float16/float32) enabled.
Loading training dataset...
  Training samples: 78750
Loading validation dataset...
  Validation samples: 16875

Initialising efficientdet_lite2 for 150 classes...
Starting training...

Epoch 1/100
4922/4922 [==============================] - 1240s 252ms/step
  - loss: 3.4521 - box_loss: 0.0412 - cls_loss: 3.4109
  - val_loss: 3.1203 - val_AP: 0.0182

Epoch 2/100
...
```

### 6.2 Training time estimates

| GPU | Batch size | Time per epoch | 100 epochs |
|-----|-----------|----------------|------------|
| RTX 3090 24GB | 16 | ~22 min | ~37 h |
| A100 40GB | 32 | ~8 min | ~13 h |
| A100 80GB | 64 | ~5 min | ~8 h |

### 6.3 Resuming from a checkpoint

If training is interrupted:

```bash
python src/train_model.py \
    --config     configs/training_config.yaml \
    --output_dir /models/efficientdet_lite2_150cls \
    --resume_checkpoint /models/efficientdet_lite2_150cls/checkpoints/checkpoint_epoch_045
```

---

## Step 7: Monitoring with TensorBoard

```bash
tensorboard --logdir /models/efficientdet_lite2_150cls/logs --port 6006
```

Open `http://localhost:6006` in a browser.

### Key metrics to watch

- **val_AP**: primary metric; should increase steadily.  Values of 0.4+ after
  100 epochs are achievable with a clean 150-class dataset.
- **val_loss**: should decrease in parallel with AP improvements.
- **box_loss vs cls_loss**: if box_loss stops decreasing but cls_loss does not,
  the model is learning class discrimination but struggling with localisation
  -- this often indicates noisy bounding-box annotations.

### 150-class specific TensorBoard tips

- Use the **Scalars** tab to filter for `val_AP` per class if your training
  script logs per-class AP.
- Watch for class groups that plateau early: fine-grained classes (e.g. dog
  breeds) typically reach lower AP than coarse classes (e.g. vehicle).

---

## Step 8: Evaluating the Model

```bash
python src/evaluate_model.py \
    --model      /models/efficientdet_lite2_150cls/best_model.tflite \
    --dataset    /data/tfrecords/test \
    --class_map  /data/tfrecords/class_map.json \
    --output_dir /reports \
    --top_n 20
```

### 8.1 Expected output

```
mAP@0.5      : 0.4312
mAP@0.5:0.95 : 0.2871
Evaluated on : 16875 images

--- TOP 20 CLASSES (by F1) ---
Class                                      GT   Prec    Rec     F1   AP50
--------------------------------------------------------------------
car                                      1830  0.921  0.884  0.902  0.891
bicycle                                  1620  0.887  0.861  0.874  0.855
...

--- BOTTOM 20 CLASSES (by F1) ---
...
```

### 8.2 Interpreting results for 150 classes

A mAP@0.5 of **0.35-0.45** is a reasonable target for 150 classes trained from
scratch with 500 images/class and good annotation quality.

Investigate bottom-20 classes:
1. Check annotation quality for those classes.
2. Verify image diversity (not all images from the same viewpoint).
3. Consider adding more training data or using copy-paste augmentation.

---

## Step 9: Exporting to TFLite

The training script exports TFLite automatically.  To re-export manually with
different quantisation settings:

### 9.1 Dynamic-range quantisation (default)

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(
    "/models/efficientdet_lite2_150cls/saved_model"
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("best_model_dynamic.tflite", "wb") as f:
    f.write(tflite_model)
print(f"Model size: {len(tflite_model) / 1024 / 1024:.1f} MB")
```

### 9.2 Full INT8 quantisation

```python
import numpy as np
import tensorflow as tf

def representative_dataset():
    ds = tf.data.TFRecordDataset(
        tf.io.gfile.glob("/data/tfrecords/val/shard-*.tfrecord")
    )
    feature_spec = {"image/encoded": tf.io.FixedLenFeature([], tf.string)}
    for i, raw in enumerate(ds.take(500)):
        parsed = tf.io.parse_single_example(raw, feature_spec)
        img = tf.io.decode_image(parsed["image/encoded"], channels=3)
        img = tf.image.resize(img, [512, 512])
        img = tf.cast(img, tf.uint8)
        yield [tf.expand_dims(img, 0)]

converter = tf.lite.TFLiteConverter.from_saved_model(
    "/models/efficientdet_lite2_150cls/saved_model"
)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()
with open("best_model_int8.tflite", "wb") as f:
    f.write(tflite_model)
```

INT8 quantisation reduces model size by ~4x and inference latency by ~2x on
ARM CPUs, with typically less than 1% mAP drop.

---

## Step 10: Integrating the Custom Model with MediaPipe

### 10.1 Add TFLite Model Metadata

MediaPipe requires metadata to be embedded in the TFLite file so it knows the
label map and normalisation parameters.

```python
from mediapipe_model_maker.python.core.utils import model_util

model_util.add_metadata(
    tflite_path="best_model.tflite",
    label_map_path="/data/tfrecords/class_map.json",
    export_path="best_model_with_metadata.tflite",
)
```

### 10.2 Python inference with MediaPipe Tasks API

```python
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BaseOptions = mp_python.BaseOptions
ObjectDetector = vision.ObjectDetector
ObjectDetectorOptions = vision.ObjectDetectorOptions
VisionRunningMode = vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path="best_model_with_metadata.tflite"),
    running_mode=VisionRunningMode.IMAGE,
    max_results=50,
    score_threshold=0.3,
)

with ObjectDetector.create_from_options(options) as detector:
    image = mp.Image.create_from_file("test_image.jpg")
    result = detector.detect(image)
    for detection in result.detections:
        category = detection.categories[0]
        print(f"{category.category_name}: {category.score:.2f} | "
              f"bbox={detection.bounding_box}")
```

### 10.3 Android integration

1. Copy `best_model_with_metadata.tflite` into `app/src/main/assets/`.
2. Add the MediaPipe Tasks dependency to `build.gradle`:

```gradle
dependencies {
    implementation 'com.google.mediapipe:tasks-vision:0.10.3'
}
```

3. Initialise in Java/Kotlin:

```kotlin
val options = ObjectDetectorOptions.builder()
    .setBaseOptions(BaseOptions.builder()
        .setModelAssetPath("best_model_with_metadata.tflite")
        .build())
    .setMaxResults(50)
    .setScoreThreshold(0.3f)
    .build()

val detector = ObjectDetector.createFromOptions(context, options)
```

---

## Troubleshooting

### "OOM: allocator (GPU_0_bfc) ran out of memory"

Reduce batch size in `configs/training_config.yaml`:

```yaml
training:
  batch_size: 8    # down from 16
```

With batch size 8 and mixed precision you need approximately 10 GB VRAM for
EfficientDet-Lite2 at 512x512.

### Training loss is NaN after epoch 1

This usually means the learning rate is too high.  Reduce it and increase warmup:

```yaml
training:
  learning_rate: 0.04
  lr_warmup_epochs: 10
```

### mAP@0.5 stuck below 0.10 after 20 epochs

1. Verify TFRecord content with `visualise_dataset.py` -- confirm boxes look correct.
2. Check class_map.json -- indices must start at 0 and be contiguous.
3. Ensure images are not all-black or all-white (data loading error).

### Certain classes have 0.0 AP

- Check that the class has at least 50 validation images.
- Verify the category ID in the COCO JSON is not 0 (reserved for background).
- Run `prepare_dataset.py` with `--validate` to see per-class sample counts.

### Export fails with "RuntimeError: delegate is not supported"

The exported TFLite model uses NNAPI delegates.  Force CPU-only export:

```python
converter.target_spec.supported_backends = [tf.lite.OpsSet.TFLITE_BUILTINS]
```

### 150 classes but only 140 appear in predictions

Check that all 150 category IDs are present in your COCO JSON
`"categories"` list.  The `prepare_dataset.py` script will log a warning for
any class with fewer than 300 training images, which may be missing entirely if
you filtered under-represented classes.

### Fine-grained classes (e.g. dog breeds) have low AP

This is expected when the visual difference between classes is subtle.
Strategies:

1. **Increase data**: collect 700-1000 images per breed, covering diverse poses,
   lighting, and backgrounds.
2. **Use a larger backbone**: switch to EfficientDet-Lite3 (640x640 input).
3. **Use copy-paste augmentation**: paste breed instances onto diverse backgrounds.
4. **Two-stage approach**: train a coarse model (dog vs. not-dog) first, then a
   fine-grained breed classifier on cropped regions.
