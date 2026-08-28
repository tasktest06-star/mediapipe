# Synthetic Training Data Pipeline for MediaPipe Object Detection

## Why Synthetic Data Matters

Training a reliable custom object detector traditionally demands hundreds or thousands of real,
hand-annotated images per class. For product catalogues with tens or hundreds of SKUs this
annotation cost is prohibitive. Synthetic data generation solves the cold-start problem by
automatically compositing product cut-outs onto diverse backgrounds, applying physics-inspired
lighting and shadow models, and producing COCO-format annotations with zero manual effort.

### What quality is achievable?

| Training data mix | Expected mAP@0.5 | Real images labelled |
|---|---|---|
| Synthetic only (1 000 imgs/class) | ~0.45 – 0.55 | 0 |
| 90% synthetic + 10% real | ~0.60 – 0.70 | ~20 |
| 50% synthetic + 50% real | ~0.72 – 0.80 | ~100 |
| 100% real (1 000 imgs/class) | ~0.80 – 0.88 | 1 000 |

The numbers above are indicative benchmarks from published domain-adaptation research.
Actual results depend heavily on the domain gap between your composited backgrounds and the
real deployment environment.

**Domain-gap sources to mitigate:**
- Unrealistic lighting (too uniform, no specular highlights)
- Missing cast shadows
- Background statistics mismatch (synthetic vs. real shelf/table textures)
- Scale distribution mismatch

This pipeline addresses all four through configurable augmentation, shadow simulation,
background diversity modules, and a domain-adaptive mixing strategy.

---

## Prerequisites

- Python 3.9 – 3.11
- 4 GB RAM minimum (8 GB recommended for large catalogues)
- Optional: CUDA-capable GPU for faster TFRecord conversion and training

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package | Purpose |
|---|---|
| `Pillow` | Image compositing and augmentation |
| `numpy` | Array operations |
| `rembg` | Automatic background removal (optional) |
| `tqdm` | Progress bars |
| `mediapipe-model-maker` | MediaPipe training API |
| `tensorflow` | TFRecord conversion and training backend |
| `pyyaml` | Config loading |
| `pycocotools` | COCO annotation utilities |

---

## Quick-Start (5 Commands)

```bash
# 1. Install dependencies
pip install Pillow numpy rembg tqdm mediapipe-model-maker tensorflow pyyaml pycocotools

# 2. Parse product catalogue and build class map
python src/product_catalog_parser.py \
    --catalog data/sample_catalog.json \
    --output data/processed/ \
    --image-dir data/product_images/

# 3. Generate synthetic training dataset
python src/generate_dataset.py \
    --config configs/generation_config.yaml \
    --catalog data/processed/class_map.json \
    --backgrounds data/backgrounds/ \
    --output data/synthetic_dataset/

# 4. Train MediaPipe object detector
python src/train_with_synthetic.py \
    --dataset data/synthetic_dataset/ \
    --real-data data/real_dataset/ \
    --mix-ratio 0.1 \
    --output models/my_detector/

# 5. The TFLite model is exported automatically at the end of step 4
ls models/my_detector/model.tflite
```

---

## Repository Layout

```
.
├── README.md
├── data/
│   ├── sample_catalog.json        # 25-product example catalogue
│   ├── class_hierarchy.json       # class tree (product -> category -> super)
│   ├── product_images/            # place raw product images here
│   ├── backgrounds/               # optional real-background images
│   └── synthetic_dataset/         # generated output (created by pipeline)
├── configs/
│   └── generation_config.yaml     # all generation parameters
├── src/
│   ├── product_catalog_parser.py
│   ├── background_generator.py
│   ├── synthetic_compositor.py
│   ├── augmentation_pipeline.py
│   ├── generate_dataset.py
│   ├── quality_validator.py
│   └── train_with_synthetic.py
└── docs/
    ├── synthetic_data_guide.md    # comprehensive written guide
    └── synthetic_data_guide.tex   # LaTeX version with TikZ diagrams
```

---

## Further Reading

- [docs/synthetic_data_guide.md](docs/synthetic_data_guide.md) - step-by-step guide with
  runnable code snippets, quality benchmarks, domain-gap analysis, and Blender integration notes.
- [MediaPipe Model Maker documentation](https://developers.google.com/mediapipe/solutions/model_maker)
