# MediaPipe EfficientDet-Lite — 150-Class Custom Training Pipeline

This branch contains a complete, production-ready pipeline for training a
MediaPipe EfficientDet-Lite2 object-detection model on **150 custom classes
from scratch**.  Every stage — dataset preparation, augmentation, training,
evaluation, and TFLite export — is covered with runnable code and
step-by-step documentation.

---

## Repository Layout

```
.
├── README.md                          <- this file
├── requirements/
│   ├── data_requirements.txt          <- dataset size, image standards, annotation rules
│   └── python_requirements.txt        <- pinned Python dependencies
├── configs/
│   └── training_config.yaml           <- full hyper-parameter and export config
├── src/
│   ├── prepare_dataset.py             <- convert COCO/VOC/YOLO to TFRecord
│   ├── train_model.py                 <- MediaPipe Model Maker training script
│   ├── evaluate_model.py              <- mAP, per-class P/R/F1, confusion matrix
│   ├── augmentation.py                <- Albumentations + Mosaic pipeline
│   └── visualise_dataset.py           <- bounding-box previews, distribution charts, PDF report
└── docs/
    ├── custom_training_guide.md       <- 3 000+ word step-by-step guide (Markdown)
    └── custom_training_guide.tex      <- LaTeX version with TikZ diagrams and booktabs tables
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | 16 GB   | 40 GB (A100) |
| System RAM | 32 GB  | 64 GB        |
| Storage   | 500 GB SSD | 1 TB NVMe SSD |
| CUDA      | 11.8   | 12.1          |

---

## Time Estimates (EfficientDet-Lite2, 150 classes, ~100 k images)

| Stage | RTX 3090 | A100 40 GB |
|-------|----------|------------|
| Dataset preparation | ~2 h | ~45 min |
| Training (100 epochs) | ~36 h | ~10 h |
| Evaluation | ~30 min | ~10 min |
| TFLite export | ~5 min | ~5 min |

---

## Quick Start (5 commands)

```bash
# 1. Install dependencies
pip install -r requirements/python_requirements.txt

# 2. Prepare dataset (COCO JSON input example)
python src/prepare_dataset.py \
    --input_format coco \
    --annotations /data/annotations/instances_all.json \
    --images_dir   /data/images \
    --output_dir   /data/tfrecords

# 3. Train the model
python src/train_model.py \
    --config configs/training_config.yaml \
    --output_dir /models/efficientdet_lite2_150cls

# 4. Evaluate
python src/evaluate_model.py \
    --model   /models/efficientdet_lite2_150cls/best_model.tflite \
    --dataset /data/tfrecords/test \
    --class_map /data/tfrecords/class_map.json

# 5. Visualise samples
python src/visualise_dataset.py \
    --tfrecord_dir /data/tfrecords/train \
    --class_map    /data/tfrecords/class_map.json \
    --output_pdf   /reports/dataset_report.pdf
```

---

## Documentation

- **Data requirements and annotation standards** -> `requirements/data_requirements.txt`
- **Step-by-step training guide** -> `docs/custom_training_guide.md`
- **Printable PDF guide** -> compile `docs/custom_training_guide.tex` with `pdflatex`

---

## Licence

Apache 2.0 -- see `LICENSE` at the repository root.
