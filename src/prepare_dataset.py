"""prepare_dataset.py — Convert COCO JSON / Pascal VOC XML / YOLO TXT annotations
to TFRecord files suitable for training an EfficientDet-Lite model with
mediapipe_model_maker.  Performs stratified train/val/test split and writes a
class_map.json index file.

Usage
-----
python prepare_dataset.py \\
    --input_format coco \\
    --annotations  /data/annotations/instances_all.json \\
    --images_dir   /data/images \\
    --output_dir   /data/tfrecords \\
    --splits 0.70 0.15 0.15 \\
    --shards 32
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    """Axis-aligned bounding box in absolute pixel coordinates."""
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def area(self) -> float:
        return max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)

    def is_valid(self, min_size: int = 32) -> bool:
        w = self.xmax - self.xmin
        h = self.ymax - self.ymin
        return w >= min_size and h >= min_size


@dataclass
class Annotation:
    """Single annotation for one object instance in an image."""
    label: str
    label_id: int          # 0-based index into class_map
    bbox: BBox
    crowd: bool = False    # true -> ignore during evaluation


@dataclass
class ImageRecord:
    """All annotations associated with one image file."""
    image_path: str
    width: int
    height: int
    annotations: List[Annotation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# COCO JSON parser
# ---------------------------------------------------------------------------

def load_coco(annotations_path: str, images_dir: str) -> Tuple[List[ImageRecord], Dict[int, str]]:
    """Load a COCO-format JSON annotation file.

    Parameters
    ----------
    annotations_path:
        Path to the COCO instances JSON file.
    images_dir:
        Root directory that contains the image files referenced by the JSON.

    Returns
    -------
    records:
        List of ImageRecord objects, one per image.
    class_map:
        Mapping from 0-based label index to string label name.
    """
    with open(annotations_path, "r") as f:
        coco = json.load(f)

    # Build category lookup: coco_id -> (0-based-index, name)
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    coco_id_to_label: Dict[int, Tuple[int, str]] = {}
    class_map: Dict[int, str] = {}
    for idx, cat in enumerate(categories):
        coco_id_to_label[cat["id"]] = (idx, cat["name"])
        class_map[idx] = cat["name"]

    # Build image lookup: image_id -> ImageRecord
    image_lookup: Dict[int, ImageRecord] = {}
    for img in coco["images"]:
        path = os.path.join(images_dir, img["file_name"])
        image_lookup[img["id"]] = ImageRecord(
            image_path=path,
            width=img["width"],
            height=img["height"],
        )

    # Attach annotations
    for ann in coco.get("annotations", []):
        if ann["image_id"] not in image_lookup:
            continue
        label_idx, label_name = coco_id_to_label.get(ann["category_id"], (-1, "unknown"))
        if label_idx == -1:
            continue
        x, y, w, h = ann["bbox"]
        bbox = BBox(xmin=x, ymin=y, xmax=x + w, ymax=y + h)
        if not bbox.is_valid():
            continue
        crowd = bool(ann.get("iscrowd", 0))
        image_lookup[ann["image_id"]].annotations.append(
            Annotation(label=label_name, label_id=label_idx, bbox=bbox, crowd=crowd)
        )

    records = [r for r in image_lookup.values() if r.annotations]
    return records, class_map


# ---------------------------------------------------------------------------
# Pascal VOC XML parser
# ---------------------------------------------------------------------------

def load_voc(images_dir: str, annotations_dir: str) -> Tuple[List[ImageRecord], Dict[int, str]]:
    """Load Pascal VOC XML annotations from a directory.

    Parameters
    ----------
    images_dir:
        Directory containing image files.
    annotations_dir:
        Directory containing one XML file per image.

    Returns
    -------
    records, class_map
    """
    label_to_id: Dict[str, int] = {}
    records: List[ImageRecord] = []

    xml_files = sorted(glob.glob(os.path.join(annotations_dir, "*.xml")))
    for xml_path in tqdm(xml_files, desc="Parsing VOC XML"):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.findtext("filename", default="")
        image_path = os.path.join(images_dir, filename)
        size_node = root.find("size")
        if size_node is None:
            continue
        width = int(size_node.findtext("width", "0"))
        height = int(size_node.findtext("height", "0"))

        record = ImageRecord(image_path=image_path, width=width, height=height)

        for obj in root.findall("object"):
            label = obj.findtext("name", "").strip()
            if not label:
                continue
            if label not in label_to_id:
                label_to_id[label] = len(label_to_id)
            label_id = label_to_id[label]
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            bbox = BBox(
                xmin=float(bndbox.findtext("xmin", "0")),
                ymin=float(bndbox.findtext("ymin", "0")),
                xmax=float(bndbox.findtext("xmax", "0")),
                ymax=float(bndbox.findtext("ymax", "0")),
            )
            if not bbox.is_valid():
                continue
            difficult = int(obj.findtext("difficult", "0"))
            record.annotations.append(
                Annotation(label=label, label_id=label_id, bbox=bbox, crowd=bool(difficult))
            )

        if record.annotations:
            records.append(record)

    class_map = {v: k for k, v in label_to_id.items()}
    return records, class_map


# ---------------------------------------------------------------------------
# YOLO TXT parser
# ---------------------------------------------------------------------------

def load_yolo(images_dir: str, labels_dir: str, names_file: str) -> Tuple[List[ImageRecord], Dict[int, str]]:
    """Load YOLO-format annotations (normalised cx cy w h per line).

    Parameters
    ----------
    images_dir:
        Directory containing image files.
    labels_dir:
        Directory containing one TXT file per image with YOLO annotations.
    names_file:
        Path to a text file listing one class name per line (0-indexed).

    Returns
    -------
    records, class_map
    """
    with open(names_file) as f:
        names = [line.strip() for line in f if line.strip()]
    class_map = {i: name for i, name in enumerate(names)}

    records: List[ImageRecord] = []
    image_exts = {".jpg", ".jpeg", ".png"}

    for img_file in tqdm(sorted(os.listdir(images_dir)), desc="Parsing YOLO labels"):
        stem, ext = os.path.splitext(img_file)
        if ext.lower() not in image_exts:
            continue
        label_file = os.path.join(labels_dir, stem + ".txt")
        if not os.path.isfile(label_file):
            continue
        image_path = os.path.join(images_dir, img_file)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            continue

        record = ImageRecord(image_path=image_path, width=width, height=height)

        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                xmin = (cx - bw / 2) * width
                ymin = (cy - bh / 2) * height
                xmax = (cx + bw / 2) * width
                ymax = (cy + bh / 2) * height
                bbox = BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
                if not bbox.is_valid():
                    continue
                label = class_map.get(cls_id, f"class_{cls_id}")
                record.annotations.append(
                    Annotation(label=label, label_id=cls_id, bbox=bbox)
                )

        if record.annotations:
            records.append(record)

    return records, class_map


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

def validate_dataset(records: List[ImageRecord], class_map: Dict[int, str],
                     min_images: int = 300, max_ratio: float = 3.0) -> None:
    """Check class representation and flag issues.

    Parameters
    ----------
    records:
        Full dataset before splitting.
    class_map:
        Index-to-name mapping.
    min_images:
        Minimum images per class to pass validation.
    max_ratio:
        Maximum allowed ratio between the largest and smallest class.
    """
    counts: Dict[int, int] = collections.Counter()
    for rec in records:
        seen = set()
        for ann in rec.annotations:
            if not ann.crowd:
                seen.add(ann.label_id)
        for lid in seen:
            counts[lid] += 1

    print("\n--- Dataset Validation ---")
    under_rep = [class_map[lid] for lid, cnt in counts.items() if cnt < min_images]
    if under_rep:
        print(f"WARNING: {len(under_rep)} classes have fewer than {min_images} images:")
        for name in sorted(under_rep):
            lid = next(k for k, v in class_map.items() if v == name)
            print(f"  {name}: {counts[lid]} images")

    if counts:
        max_cnt = max(counts.values())
        min_cnt = min(counts.values())
        ratio = max_cnt / max(min_cnt, 1)
        if ratio > max_ratio:
            print(f"WARNING: Class imbalance ratio {ratio:.1f}:1 exceeds limit {max_ratio}:1")
        else:
            print(f"Class imbalance ratio: {ratio:.1f}:1 (within {max_ratio}:1 limit) [OK]")

    print(f"Total images: {len(records)}")
    print(f"Total classes: {len(counts)}")
    print("--------------------------\n")

    # Print statistics table
    rows = sorted(counts.items(), key=lambda x: -x[1])
    print(f"{'Class':<40} {'Images':>8}")
    print("-" * 50)
    for lid, cnt in rows:
        name = class_map.get(lid, f"id_{lid}")
        print(f"{name:<40} {cnt:>8}")
    print()


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    records: List[ImageRecord],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[ImageRecord], List[ImageRecord], List[ImageRecord]]:
    """Split records into train / val / test preserving class distribution.

    Parameters
    ----------
    records:
        Full list of ImageRecord objects.
    train_frac, val_frac:
        Fractions; test_frac = 1 - train_frac - val_frac.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    train_records, val_records, test_records
    """
    test_frac = 1.0 - train_frac - val_frac

    # Assign each record its dominant class (most frequent label)
    def dominant_label(rec: ImageRecord) -> int:
        cnt: Dict[int, int] = collections.Counter(
            ann.label_id for ann in rec.annotations if not ann.crowd
        )
        return cnt.most_common(1)[0][0] if cnt else 0

    labels = [dominant_label(r) for r in records]

    train_idx, temp_idx = train_test_split(
        list(range(len(records))),
        test_size=(val_frac + test_frac),
        stratify=labels,
        random_state=seed,
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_size_relative = val_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_size_relative),
        stratify=temp_labels,
        random_state=seed,
    )

    return (
        [records[i] for i in train_idx],
        [records[i] for i in val_idx],
        [records[i] for i in test_idx],
    )


# ---------------------------------------------------------------------------
# TFRecord serialisation
# ---------------------------------------------------------------------------

def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(values: List[float]) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _int64_feature(values: List[int]) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def record_to_tf_example(rec: ImageRecord) -> Optional[tf.train.Example]:
    """Encode an ImageRecord as a tf.train.Example proto.

    Returns None if the image file cannot be read.
    """
    try:
        with open(rec.image_path, "rb") as f:
            encoded_image = f.read()
    except OSError:
        return None

    ext = Path(rec.image_path).suffix.lower()
    image_format = b"jpeg" if ext in {".jpg", ".jpeg"} else b"png"

    xmins, ymins, xmaxs, ymaxs, labels, label_ids = [], [], [], [], [], []
    for ann in rec.annotations:
        if ann.crowd:
            continue
        xmins.append(ann.bbox.xmin / rec.width)
        ymins.append(ann.bbox.ymin / rec.height)
        xmaxs.append(ann.bbox.xmax / rec.width)
        ymaxs.append(ann.bbox.ymax / rec.height)
        labels.append(ann.label.encode("utf-8"))
        label_ids.append(ann.label_id)

    feature = {
        "image/height":            _int64_feature([rec.height]),
        "image/width":             _int64_feature([rec.width]),
        "image/filename":          _bytes_feature(rec.image_path.encode("utf-8")),
        "image/source_id":         _bytes_feature(rec.image_path.encode("utf-8")),
        "image/encoded":           _bytes_feature(encoded_image),
        "image/format":            _bytes_feature(image_format),
        "image/object/bbox/xmin":  _float_feature(xmins),
        "image/object/bbox/xmax":  _float_feature(xmaxs),
        "image/object/bbox/ymin":  _float_feature(ymins),
        "image/object/bbox/ymax":  _float_feature(ymaxs),
        "image/object/class/text": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=labels)
        ),
        "image/object/class/label": _int64_feature(label_ids),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecords(
    records: List[ImageRecord],
    output_dir: str,
    split_name: str,
    num_shards: int = 32,
) -> None:
    """Write a list of ImageRecord objects to sharded TFRecord files.

    Parameters
    ----------
    records:
        Records for this split.
    output_dir:
        Parent directory; shard files go into output_dir/split_name/.
    split_name:
        One of 'train', 'val', 'test'.
    num_shards:
        Number of TFRecord shard files to create.
    """
    shard_dir = os.path.join(output_dir, split_name)
    os.makedirs(shard_dir, exist_ok=True)

    shard_size = math.ceil(len(records) / num_shards)
    written = 0
    for shard_idx in range(num_shards):
        shard_records = records[shard_idx * shard_size: (shard_idx + 1) * shard_size]
        if not shard_records:
            break
        shard_path = os.path.join(shard_dir, f"shard-{shard_idx:05d}-of-{num_shards:05d}.tfrecord")
        with tf.io.TFRecordWriter(shard_path) as writer:
            for rec in tqdm(shard_records, desc=f"{split_name} shard {shard_idx}", leave=False):
                example = record_to_tf_example(rec)
                if example is not None:
                    writer.write(example.SerializeToString())
                    written += 1

    print(f"  [{split_name}] {written} records written to {shard_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TFRecord dataset for EfficientDet-Lite training.")
    parser.add_argument("--input_format", choices=["coco", "voc", "yolo"], required=True)
    parser.add_argument("--annotations", help="COCO JSON file path (coco mode)")
    parser.add_argument("--annotations_dir", help="VOC XML directory (voc mode)")
    parser.add_argument("--labels_dir", help="YOLO label directory (yolo mode)")
    parser.add_argument("--names_file", help="YOLO class names file (yolo mode)")
    parser.add_argument("--images_dir", required=True, help="Directory containing image files")
    parser.add_argument("--output_dir", required=True, help="Output directory for TFRecords")
    parser.add_argument("--splits", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--shards", type=int, default=32, help="Number of TFRecord shards per split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load dataset
    print(f"Loading {args.input_format} dataset...")
    if args.input_format == "coco":
        if not args.annotations:
            parser.error("--annotations required for coco format")
        records, class_map = load_coco(args.annotations, args.images_dir)
    elif args.input_format == "voc":
        if not args.annotations_dir:
            parser.error("--annotations_dir required for voc format")
        records, class_map = load_voc(args.images_dir, args.annotations_dir)
    else:  # yolo
        if not args.labels_dir or not args.names_file:
            parser.error("--labels_dir and --names_file required for yolo format")
        records, class_map = load_yolo(args.images_dir, args.labels_dir, args.names_file)

    validate_dataset(records, class_map)

    # Write class_map.json
    os.makedirs(args.output_dir, exist_ok=True)
    class_map_path = os.path.join(args.output_dir, "class_map.json")
    with open(class_map_path, "w") as f:
        json.dump({str(k): v for k, v in class_map.items()}, f, indent=2)
    print(f"Class map written to {class_map_path}")

    # Split
    train_frac, val_frac, _ = args.splits
    train_recs, val_recs, test_recs = stratified_split(
        records, train_frac=train_frac, val_frac=val_frac, seed=args.seed
    )
    print(f"Split sizes: train={len(train_recs)}, val={len(val_recs)}, test={len(test_recs)}")

    # Write TFRecords
    for split_name, split_recs in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        write_tfrecords(split_recs, args.output_dir, split_name, args.shards)

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()
