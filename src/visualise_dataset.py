"""visualise_dataset.py — Visualise random samples from TFRecord shards.

Outputs
-------
- Multi-page PDF with bounding-box previews
- Class distribution bar chart
- Image size distribution scatter plot

Usage
-----
python visualise_dataset.py \\
    --tfrecord_dir /data/tfrecords/train \\
    --class_map    /data/tfrecords/class_map.json \\
    --output_pdf   /reports/dataset_report.pdf \\
    --num_samples  30
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import tensorflow as tf
from tqdm import tqdm


# ---------------------------------------------------------------------------
# TFRecord parsing (mirrors prepare_dataset.py feature spec)
# ---------------------------------------------------------------------------

FEATURE_SPEC = {
    "image/encoded":             tf.io.FixedLenFeature([], tf.string),
    "image/height":              tf.io.FixedLenFeature([], tf.int64),
    "image/width":               tf.io.FixedLenFeature([], tf.int64),
    "image/object/bbox/ymin":    tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/xmin":    tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymax":    tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/xmax":    tf.io.VarLenFeature(tf.float32),
    "image/object/class/label":  tf.io.VarLenFeature(tf.int64),
    "image/object/class/text":   tf.io.VarLenFeature(tf.string),
}


def _decode(serialised: bytes) -> Dict:
    """Decode a single TFRecord example."""
    parsed = tf.io.parse_single_example(serialised, FEATURE_SPEC)
    image  = tf.io.decode_image(parsed["image/encoded"], channels=3, expand_animations=False)
    h = int(parsed["image/height"].numpy())
    w = int(parsed["image/width"].numpy())
    ymins  = tf.sparse.to_dense(parsed["image/object/bbox/ymin"]).numpy().tolist()
    xmins  = tf.sparse.to_dense(parsed["image/object/bbox/xmin"]).numpy().tolist()
    ymaxs  = tf.sparse.to_dense(parsed["image/object/bbox/ymax"]).numpy().tolist()
    xmaxs  = tf.sparse.to_dense(parsed["image/object/bbox/xmax"]).numpy().tolist()
    labels = tf.sparse.to_dense(parsed["image/object/class/label"]).numpy().tolist()
    texts  = [t.decode("utf-8") for t in tf.sparse.to_dense(parsed["image/object/class/text"]).numpy().tolist()]
    return {
        "image":  image.numpy(),
        "height": h,
        "width":  w,
        "bboxes": list(zip(xmins, ymins, xmaxs, ymaxs)),  # normalised
        "labels": labels,
        "texts":  texts,
    }


def load_examples(tfrecord_dir: str, max_examples: int = 5000) -> List[Dict]:
    """Load up to max_examples decoded examples from TFRecord shards.

    Parameters
    ----------
    tfrecord_dir:
        Directory containing *.tfrecord shard files.
    max_examples:
        Maximum number of examples to load (for speed).

    Returns
    -------
    List of decoded example dicts.
    """
    files = sorted(glob.glob(os.path.join(tfrecord_dir, "*.tfrecord")))
    if not files:
        raise FileNotFoundError(f"No .tfrecord files in {tfrecord_dir}")
    dataset = tf.data.TFRecordDataset(files)
    examples = []
    for raw in tqdm(dataset.take(max_examples), desc="Loading TFRecords", total=max_examples):
        examples.append(_decode(raw.numpy()))
    return examples


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

def _class_colours(num_classes: int) -> List[Tuple[float, float, float]]:
    """Generate visually distinct colours for up to num_classes classes."""
    cmap = plt.get_cmap("tab20", num_classes)
    return [cmap(i)[:3] for i in range(num_classes)]


# ---------------------------------------------------------------------------
# Visualisation pages
# ---------------------------------------------------------------------------

def draw_bboxes_page(
    examples: List[Dict],
    class_map: Dict[int, str],
    num_samples: int,
    pdf: PdfPages,
) -> None:
    """Draw random sample images with overlaid bounding boxes.

    Parameters
    ----------
    examples:
        Full list of loaded examples.
    class_map:
        Index -> name mapping.
    num_samples:
        Number of images to show.
    pdf:
        Open PdfPages object to write into.
    """
    colours = _class_colours(max(class_map.keys()) + 1)
    sample = random.sample(examples, min(num_samples, len(examples)))

    images_per_row = 4
    rows = (len(sample) + images_per_row - 1) // images_per_row
    fig, axes = plt.subplots(rows, images_per_row, figsize=(20, rows * 5))
    axes = np.array(axes).flatten()

    for idx, ex in enumerate(sample):
        ax = axes[idx]
        img = ex["image"]
        h, w = img.shape[:2]
        ax.imshow(img)
        for (xmin_n, ymin_n, xmax_n, ymax_n), lbl in zip(ex["bboxes"], ex["labels"]):
            xmin = xmin_n * w
            ymin = ymin_n * h
            xmax = xmax_n * w
            ymax = ymax_n * h
            colour = colours[lbl % len(colours)]
            rect = mpatches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=1.5, edgecolor=colour, facecolor="none",
            )
            ax.add_patch(rect)
            name = class_map.get(lbl, f"cls_{lbl}")
            ax.text(xmin, max(ymin - 3, 0), name[:15],
                    color="white", fontsize=5,
                    bbox=dict(facecolor=colour, alpha=0.7, pad=1))
        ax.axis("off")
        ax.set_title(f"{w}x{h}", fontsize=6)

    for idx in range(len(sample), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Random Dataset Samples with Ground-Truth Boxes", fontsize=14)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def class_distribution_page(
    examples: List[Dict],
    class_map: Dict[int, str],
    pdf: PdfPages,
) -> None:
    """Plot a horizontal bar chart of instance counts per class.

    Parameters
    ----------
    examples:
        Full list of loaded examples.
    class_map:
        Index -> name mapping.
    pdf:
        Open PdfPages object.
    """
    counts: Dict[int, int] = {}
    for ex in examples:
        for lbl in ex["labels"]:
            counts[lbl] = counts.get(lbl, 0) + 1

    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    names  = [class_map.get(k, f"cls_{k}") for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig_height = max(8, len(names) * 0.22)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    y_pos = range(len(names))
    bars = ax.barh(y_pos, values, color=plt.cm.viridis(np.linspace(0, 0.8, len(names))))
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Instance count")
    ax.set_title("Class Distribution (instances per class)")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=5)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def image_size_distribution_page(examples: List[Dict], pdf: PdfPages) -> None:
    """Scatter plot of image widths vs heights.

    Parameters
    ----------
    examples:
        Full list of loaded examples.
    pdf:
        Open PdfPages object.
    """
    widths  = [ex["width"]  for ex in examples]
    heights = [ex["height"] for ex in examples]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(widths, heights, alpha=0.3, s=5, color="steelblue")
    ax.set_xlabel("Image width (px)")
    ax.set_ylabel("Image height (px)")
    ax.set_title(f"Image Size Distribution (n={len(examples)})")

    # Annotate aspect-ratio lines
    for ratio, label in [(4 / 3, "4:3"), (16 / 9, "16:9"), (1.0, "1:1")]:
        max_w = max(widths) if widths else 1920
        ax.plot([0, max_w], [0, max_w / ratio], "--", linewidth=0.8, label=label)
    ax.legend(fontsize=8)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise TFRecord dataset samples.")
    parser.add_argument("--tfrecord_dir", required=True, help="Directory with TFRecord shards")
    parser.add_argument("--class_map",    required=True, help="Path to class_map.json")
    parser.add_argument("--output_pdf",   default="./dataset_report.pdf", help="Output PDF path")
    parser.add_argument("--num_samples",  type=int, default=30, help="Images to show with boxes")
    parser.add_argument("--max_load",     type=int, default=5000, help="Max records to load for charts")
    args = parser.parse_args()

    with open(args.class_map) as f:
        raw = json.load(f)
    class_map = {int(k): v for k, v in raw.items()}

    print(f"Loading examples from {args.tfrecord_dir} ...")
    examples = load_examples(args.tfrecord_dir, max_examples=args.max_load)
    print(f"Loaded {len(examples)} examples.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_pdf)), exist_ok=True)
    with PdfPages(args.output_pdf) as pdf:
        print("Page 1: bounding-box previews ...")
        draw_bboxes_page(examples, class_map, args.num_samples, pdf)
        print("Page 2: class distribution chart ...")
        class_distribution_page(examples, class_map, pdf)
        print("Page 3: image size scatter plot ...")
        image_size_distribution_page(examples, pdf)

    print(f"\nReport saved to {args.output_pdf}")


if __name__ == "__main__":
    main()
