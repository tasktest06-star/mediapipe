"""evaluate_model.py — Evaluate a TFLite EfficientDet-Lite model on the test split.

Computes:
  - mAP@0.5 and mAP@0.5:0.95
  - Per-class precision, recall, and F1 (printed for top-20 and bottom-20 classes)
  - Confusion matrix saved as a PNG

Usage
-----
python evaluate_model.py \\
    --model   /models/best_model.tflite \\
    --dataset /data/tfrecords/test \\
    --class_map /data/tfrecords/class_map.json \\
    --output_dir /reports
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

import tensorflow as tf


# ---------------------------------------------------------------------------
# TFLite inference wrapper
# ---------------------------------------------------------------------------

class TFLiteDetector:
    """Wraps a TFLite object-detection model for inference.

    Parameters
    ----------
    model_path:
        Path to the .tflite file.
    num_threads:
        Number of CPU threads for inference.
    """

    def __init__(self, model_path: str, num_threads: int = 4) -> None:
        self.interpreter = Interpreter(
            model_path=model_path,
            num_threads=num_threads,
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        input_shape = self.input_details[0]["shape"]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Run inference on a single image.

        Parameters
        ----------
        image:
            BGR or RGB uint8 array (H x W x 3); will be resized automatically.

        Returns
        -------
        boxes:
            (N, 4) float32 in [ymin, xmin, ymax, xmax] normalised to [0, 1].
        classes:
            (N,) float32 class indices (0-based).
        scores:
            (N,) float32 confidence scores.
        num_detections:
            Number of valid detections.
        """
        import cv2
        resized = cv2.resize(image, (self.input_width, self.input_height))
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        boxes   = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        scores  = self.interpreter.get_tensor(self.output_details[2]["index"])[0]
        count   = int(self.interpreter.get_tensor(self.output_details[3]["index"])[0])

        return boxes, classes, scores, count


# ---------------------------------------------------------------------------
# TFRecord decoding
# ---------------------------------------------------------------------------

FEATURE_SPEC = {
    "image/encoded":            tf.io.FixedLenFeature([], tf.string),
    "image/height":             tf.io.FixedLenFeature([], tf.int64),
    "image/width":              tf.io.FixedLenFeature([], tf.int64),
    "image/object/bbox/ymin":   tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/xmin":   tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/ymax":   tf.io.VarLenFeature(tf.float32),
    "image/object/bbox/xmax":   tf.io.VarLenFeature(tf.float32),
    "image/object/class/label": tf.io.VarLenFeature(tf.int64),
}


def decode_example(serialised: bytes) -> Dict:
    """Decode a single TFRecord example to a dict of numpy arrays."""
    parsed = tf.io.parse_single_example(serialised, FEATURE_SPEC)
    image  = tf.io.decode_image(parsed["image/encoded"], channels=3, expand_animations=False)
    gt_boxes = tf.stack([
        tf.sparse.to_dense(parsed["image/object/bbox/ymin"]),
        tf.sparse.to_dense(parsed["image/object/bbox/xmin"]),
        tf.sparse.to_dense(parsed["image/object/bbox/ymax"]),
        tf.sparse.to_dense(parsed["image/object/bbox/xmax"]),
    ], axis=1)
    gt_labels = tf.cast(tf.sparse.to_dense(parsed["image/object/class/label"]), tf.int32)
    return {
        "image":     image.numpy(),
        "gt_boxes":  gt_boxes.numpy(),
        "gt_labels": gt_labels.numpy(),
    }


def load_tfrecords(dataset_dir: str):
    """Yield decoded examples from all TFRecord shards in dataset_dir."""
    pattern = os.path.join(dataset_dir, "*.tfrecord")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No TFRecord files found in {dataset_dir}")
    raw_dataset = tf.data.TFRecordDataset(files)
    for serialised in raw_dataset:
        yield decode_example(serialised.numpy())


# ---------------------------------------------------------------------------
# IoU and matching utilities
# ---------------------------------------------------------------------------

def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes in [ymin, xmin, ymax, xmax] format."""
    y1 = max(box_a[0], box_b[0])
    x1 = max(box_a[1], box_b[1])
    y2 = min(box_a[2], box_b[2])
    x2 = min(box_a[3], box_b[3])
    inter = max(0, y2 - y1) * max(0, x2 - x1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute AP using the 101-point interpolation (COCO style)."""
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        prec_at_t = precisions[recalls >= t]
        ap += (np.max(prec_at_t) if prec_at_t.size > 0 else 0.0)
    return ap / 101.0


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def evaluate(
    detector: TFLiteDetector,
    dataset_dir: str,
    class_map: Dict[int, str],
    iou_thresholds: List[float],
    score_threshold: float = 0.3,
    max_detections: int = 100,
) -> Dict:
    """Run full evaluation over the test split.

    Parameters
    ----------
    detector:
        Loaded TFLiteDetector instance.
    dataset_dir:
        Directory containing test TFRecord shards.
    class_map:
        Index -> label name mapping.
    iou_thresholds:
        List of IoU thresholds to evaluate at.
    score_threshold:
        Discard detections below this confidence.
    max_detections:
        Maximum detections to consider per image.

    Returns
    -------
    results:
        Dict with 'per_class' and 'map' keys.
    """
    num_classes = max(class_map.keys()) + 1
    # per-class: list of (score, tp) tuples and gt counts
    detections: Dict[int, List[Tuple[float, int]]] = collections.defaultdict(list)
    gt_counts: Dict[int, int] = collections.Counter()
    # confusion: predicted_class -> true_class (for TP matches at IoU=0.5)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int32)

    examples = list(load_tfrecords(dataset_dir))
    for ex in tqdm(examples, desc="Evaluating"):
        image  = ex["image"]
        gt_boxes  = ex["gt_boxes"]
        gt_labels = ex["gt_labels"]

        for lbl in gt_labels:
            gt_counts[int(lbl)] += 1

        boxes, classes, scores, count = detector.predict(image)

        # Keep top-max_detections by score
        order = np.argsort(-scores[:count])[:max_detections]
        pred_boxes   = boxes[order]
        pred_classes = classes[order].astype(int)
        pred_scores  = scores[order]

        matched_gt = set()
        for pred_idx in range(len(order)):
            if pred_scores[pred_idx] < score_threshold:
                continue
            cls_id = pred_classes[pred_idx]
            best_iou, best_gt = 0.0, -1
            for gt_idx, (gt_box, gt_lbl) in enumerate(zip(gt_boxes, gt_labels)):
                if gt_idx in matched_gt:
                    continue
                if int(gt_lbl) != cls_id:
                    continue
                iou_val = iou(pred_boxes[pred_idx], gt_box)
                if iou_val > best_iou:
                    best_iou, best_gt = iou_val, gt_idx
            # IoU=0.5 threshold for TP
            if best_iou >= 0.5 and best_gt != -1:
                matched_gt.add(best_gt)
                detections[cls_id].append((pred_scores[pred_idx], 1))
                confusion[cls_id, cls_id] += 1
            else:
                detections[cls_id].append((pred_scores[pred_idx], 0))
                # Find what the gt was (for confusion)
                if best_gt != -1:
                    true_cls = int(gt_labels[best_gt])
                    confusion[cls_id, true_cls] += 1

    # Compute per-class AP at each IoU threshold
    per_class_ap: Dict[int, Dict[float, float]] = {}
    for cls_id in range(num_classes):
        per_class_ap[cls_id] = {}
        cls_dets = sorted(detections[cls_id], key=lambda x: -x[0])
        tp_arr = np.array([d[1] for d in cls_dets])
        fp_arr = 1 - tp_arr
        tp_cum = np.cumsum(tp_arr)
        fp_cum = np.cumsum(fp_arr)
        n_gt = gt_counts.get(cls_id, 0)
        if n_gt == 0:
            for thr in iou_thresholds:
                per_class_ap[cls_id][thr] = float("nan")
            continue
        recalls    = tp_cum / n_gt
        precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
        for thr in iou_thresholds:
            per_class_ap[cls_id][thr] = compute_ap(recalls, precisions)

    # mAP@0.5 and mAP@0.5:0.95
    valid_aps_05   = [per_class_ap[c][0.5]                   for c in range(num_classes)
                      if not np.isnan(per_class_ap[c][0.5])]
    valid_aps_5095 = [np.nanmean([per_class_ap[c][t] for t in iou_thresholds])
                      for c in range(num_classes)
                      if not np.isnan(per_class_ap[c][iou_thresholds[0]])]

    map_05   = float(np.mean(valid_aps_05))   if valid_aps_05   else 0.0
    map_5095 = float(np.mean(valid_aps_5095)) if valid_aps_5095 else 0.0

    # Per-class precision, recall, F1 at IoU=0.5, score_threshold
    per_class_metrics = {}
    for cls_id in range(num_classes):
        cls_dets = [(s, tp) for s, tp in detections[cls_id] if s >= score_threshold]
        tp = sum(tp for _, tp in cls_dets)
        fp = len(cls_dets) - tp
        fn = max(0, gt_counts.get(cls_id, 0) - tp)
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        per_class_metrics[cls_id] = {
            "name":      class_map.get(cls_id, f"class_{cls_id}"),
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "ap50":      per_class_ap[cls_id].get(0.5, float("nan")),
            "gt_count":  gt_counts.get(cls_id, 0),
        }

    return {
        "map_05":          map_05,
        "map_5095":        map_5095,
        "per_class":       per_class_metrics,
        "confusion_matrix": confusion,
        "num_images":      len(examples),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: Dict, top_n: int = 20) -> None:
    """Print per-class metrics for the top-N and bottom-N classes by F1."""
    metrics = [v for v in results["per_class"].values() if v["gt_count"] > 0]
    sorted_by_f1 = sorted(metrics, key=lambda x: -x["f1"])

    header = f"{'Class':<40} {'GT':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AP50':>6}"
    sep = "-" * 68

    print(f"\nmAP@0.5      : {results['map_05']:.4f}")
    print(f"mAP@0.5:0.95 : {results['map_5095']:.4f}")
    print(f"Evaluated on : {results['num_images']} images\n")

    print("--- TOP 20 CLASSES (by F1) ---")
    print(header)
    print(sep)
    for m in sorted_by_f1[:top_n]:
        ap50 = f"{m['ap50']:.3f}" if not np.isnan(m["ap50"]) else "  N/A"
        print(f"{m['name']:<40} {m['gt_count']:>6} {m['precision']:>6.3f} "
              f"{m['recall']:>6.3f} {m['f1']:>6.3f} {ap50:>6}")

    print("\n--- BOTTOM 20 CLASSES (by F1) ---")
    print(header)
    print(sep)
    for m in sorted_by_f1[-top_n:]:
        ap50 = f"{m['ap50']:.3f}" if not np.isnan(m["ap50"]) else "  N/A"
        print(f"{m['name']:<40} {m['gt_count']:>6} {m['precision']:>6.3f} "
              f"{m['recall']:>6.3f} {m['f1']:>6.3f} {ap50:>6}")


def save_confusion_matrix(confusion: np.ndarray, class_map: Dict[int, str], output_path: str) -> None:
    """Save confusion matrix as a PNG image.

    Parameters
    ----------
    confusion:
        NxN integer array.
    class_map:
        Index-to-name mapping.
    output_path:
        Destination PNG file path.
    """
    n = confusion.shape[0]
    # Only plot classes that have at least one GT instance
    active = [i for i in range(n) if confusion[i].sum() + confusion[:, i].sum() > 0]
    active = active[:80]  # cap at 80 to keep figure readable
    sub = confusion[np.ix_(active, active)]
    labels = [class_map.get(i, str(i))[:12] for i in active]

    fig_size = max(12, len(active) // 3)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    cax = ax.imshow(sub, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax)
    ax.set_xticks(range(len(active)))
    ax.set_yticks(range(len(active)))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("True class")
    ax.set_ylabel("Predicted class")
    ax.set_title("Confusion Matrix (top 80 active classes)")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TFLite EfficientDet model.")
    parser.add_argument("--model",       required=True, help="Path to .tflite file")
    parser.add_argument("--dataset",     required=True, help="Directory with test TFRecord shards")
    parser.add_argument("--class_map",   required=True, help="Path to class_map.json")
    parser.add_argument("--output_dir",  default="./reports", help="Where to save confusion matrix PNG")
    parser.add_argument("--score_threshold", type=float, default=0.3)
    parser.add_argument("--max_detections",  type=int,   default=100)
    parser.add_argument("--top_n",           type=int,   default=20,
                        help="Number of top/bottom classes to print")
    parser.add_argument("--num_threads",     type=int,   default=4)
    args = parser.parse_args()

    with open(args.class_map) as f:
        raw = json.load(f)
    class_map = {int(k): v for k, v in raw.items()}

    print(f"Loading model from {args.model} ...")
    detector = TFLiteDetector(args.model, num_threads=args.num_threads)

    iou_thresholds = [round(0.5 + i * 0.05, 2) for i in range(10)]  # 0.50 to 0.95

    results = evaluate(
        detector=detector,
        dataset_dir=args.dataset,
        class_map=class_map,
        iou_thresholds=iou_thresholds,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
    )

    print_table(results, top_n=args.top_n)

    confusion_path = os.path.join(args.output_dir, "confusion_matrix.png")
    save_confusion_matrix(results["confusion_matrix"], class_map, confusion_path)

    # Save JSON summary
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        "map_05":     results["map_05"],
        "map_5095":   results["map_5095"],
        "num_images": results["num_images"],
        "per_class": {
            str(k): {kk: (float(vv) if not isinstance(vv, str) else vv)
                     for kk, vv in v.items()}
            for k, v in results["per_class"].items()
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Evaluation summary saved to {summary_path}")


if __name__ == "__main__":
    main()
