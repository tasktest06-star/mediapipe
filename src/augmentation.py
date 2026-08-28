"""augmentation.py — Albumentations-based augmentation pipeline with bounding-box
support for EfficientDet-Lite training.

Features
--------
- HorizontalFlip, BrightnessContrast, HueSaturation
- RandomCrop, ShiftScaleRotate
- Mosaic 4-image augmentation (manual implementation)
- Cutout / CoarseDropout

Usage
-----
from augmentation import Augmenter
aug = Augmenter(image_size=512, mosaic_prob=0.5)
aug_image, aug_bboxes, aug_labels = aug.apply(image, bboxes, labels)
"""

from __future__ import annotations

import random
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
# bbox: [xmin, ymin, xmax, ymax] in absolute pixels (not normalised)
BBoxes = List[List[float]]   # [[xmin, ymin, xmax, ymax], ...]
Labels = List[int]           # corresponding class indices


# ---------------------------------------------------------------------------
# Helper: build the Albumentations transform pipeline
# ---------------------------------------------------------------------------

def _build_pipeline(image_size: int) -> A.Compose:
    """Create the standard augmentation pipeline.

    Parameters
    ----------
    image_size:
        Square target size for the output image (e.g. 512 for EfficientDet-Lite2).

    Returns
    -------
    transform:
        Albumentations Compose object configured to transform bounding boxes.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=0.4,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.RandomCrop(
                height=int(image_size * 0.85),
                width=int(image_size * 0.85),
                p=0.3,
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=64,
                max_width=64,
                min_holes=1,
                min_height=16,
                min_width=16,
                fill_value=0,
                p=0.3,
            ),
            A.Resize(height=image_size, width=image_size, p=1.0),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",        # [xmin, ymin, xmax, ymax] absolute
            label_fields=["class_ids"],
            min_area=32 * 32,           # drop boxes smaller than 32x32 after aug
            min_visibility=0.3,         # drop boxes with < 30% visibility after crop
        ),
    )


# ---------------------------------------------------------------------------
# Mosaic implementation
# ---------------------------------------------------------------------------

def mosaic_4(
    images: List[np.ndarray],
    bboxes_list: List[BBoxes],
    labels_list: List[Labels],
    target_size: int,
) -> Tuple[np.ndarray, BBoxes, Labels]:
    """Combine four images into a single mosaic image.

    Each image is placed in one quadrant of a (2*target_size x 2*target_size)
    canvas, then the canvas is cropped back to (target_size x target_size) around
    a random centre point.  Bounding boxes are transformed accordingly.

    Parameters
    ----------
    images:
        List of exactly 4 BGR/RGB images (any size; will be resized to target_size).
    bboxes_list:
        List of 4 bbox lists, each in absolute [xmin, ymin, xmax, ymax] format.
    labels_list:
        List of 4 label lists corresponding to bboxes_list.
    target_size:
        Square output size.

    Returns
    -------
    mosaic_img, merged_bboxes, merged_labels
    """
    assert len(images) == 4, "mosaic_4 requires exactly 4 images"
    s = target_size

    # Choose a random centre within [0.25s, 0.75s]
    cx = int(random.uniform(0.25 * s, 0.75 * s))
    cy = int(random.uniform(0.25 * s, 0.75 * s))

    canvas = np.zeros((2 * s, 2 * s, 3), dtype=np.uint8)

    # Quadrant offsets: (x_start, y_start) for each of the 4 images
    # top-left, top-right, bottom-left, bottom-right
    placements = [
        (s - cx, s - cy),   # image 0 -> top-left quadrant
        (s,      s - cy),   # image 1 -> top-right quadrant
        (s - cx, s),        # image 2 -> bottom-left quadrant
        (s,      s),        # image 3 -> bottom-right quadrant
    ]
    quad_sizes = [
        (cx, cy),            # w, h for image 0
        (s - cx, cy),        # image 1
        (cx, s - cy),        # image 2
        (s - cx, s - cy),    # image 3
    ]

    merged_bboxes: BBoxes = []
    merged_labels: Labels = []

    for idx in range(4):
        img = cv2.resize(images[idx], (s, s))
        qw, qh = quad_sizes[idx]
        x_offset, y_offset = placements[idx]

        # Source region in the resized image
        src_x = 0 if idx in (0, 2) else (s - qw)
        src_y = 0 if idx in (0, 1) else (s - qh)

        canvas[y_offset: y_offset + qh, x_offset: x_offset + qw] = (
            img[src_y: src_y + qh, src_x: src_x + qw]
        )

        for box, lbl in zip(bboxes_list[idx], labels_list[idx]):
            # Scale box from original image to resized s x s
            orig_h, orig_w = images[idx].shape[:2]
            scale_x = s / orig_w
            scale_y = s / orig_h
            bxmin = box[0] * scale_x
            bymin = box[1] * scale_y
            bxmax = box[2] * scale_x
            bymax = box[3] * scale_y

            # Shift to the canvas coordinate frame
            bxmin += x_offset - src_x
            bymin += y_offset - src_y
            bxmax += x_offset - src_x
            bymax += y_offset - src_y

            # Clip to canvas
            bxmin = np.clip(bxmin, x_offset, x_offset + qw)
            bymin = np.clip(bymin, y_offset, y_offset + qh)
            bxmax = np.clip(bxmax, x_offset, x_offset + qw)
            bymax = np.clip(bymax, y_offset, y_offset + qh)

            if (bxmax - bxmin) >= 32 and (bymax - bymin) >= 32:
                merged_bboxes.append([bxmin, bymin, bxmax, bymax])
                merged_labels.append(lbl)

    # Crop canvas to [cy : cy+s, cx : cx+s]
    cropped = canvas[cy: cy + s, cx: cx + s]

    # Adjust box coordinates relative to crop
    final_bboxes: BBoxes = []
    final_labels: Labels = []
    for box, lbl in zip(merged_bboxes, merged_labels):
        bxmin = np.clip(box[0] - cx, 0, s)
        bymin = np.clip(box[1] - cy, 0, s)
        bxmax = np.clip(box[2] - cx, 0, s)
        bymax = np.clip(box[3] - cy, 0, s)
        if (bxmax - bxmin) >= 32 and (bymax - bymin) >= 32:
            final_bboxes.append([bxmin, bymin, bxmax, bymax])
            final_labels.append(lbl)

    return cropped, final_bboxes, final_labels


# ---------------------------------------------------------------------------
# Main Augmenter class
# ---------------------------------------------------------------------------

class Augmenter:
    """Apply augmentation to a single image with its bounding boxes.

    Parameters
    ----------
    image_size:
        Square output image size (pixels).
    mosaic_prob:
        Probability of applying mosaic augmentation when a batch of 4
        images is provided via apply_mosaic().
    """

    def __init__(self, image_size: int = 512, mosaic_prob: float = 0.5) -> None:
        self.image_size = image_size
        self.mosaic_prob = mosaic_prob
        self._pipeline = _build_pipeline(image_size)

    def apply(
        self,
        image: np.ndarray,
        bboxes: BBoxes,
        labels: Labels,
    ) -> Tuple[np.ndarray, BBoxes, Labels]:
        """Apply the standard (non-mosaic) augmentation pipeline.

        Parameters
        ----------
        image:
            H x W x 3 uint8 image array (BGR or RGB; colour transforms are
            channel-agnostic, so either convention is fine).
        bboxes:
            List of [xmin, ymin, xmax, ymax] bounding boxes in absolute pixels.
        labels:
            Integer class indices corresponding to each box.

        Returns
        -------
        aug_image, aug_bboxes, aug_labels
            Augmented image (target_size x target_size x 3) and updated boxes/labels.
        """
        result = self._pipeline(image=image, bboxes=bboxes, class_ids=labels)
        return result["image"], list(result["bboxes"]), list(result["class_ids"])

    def apply_mosaic(
        self,
        images: List[np.ndarray],
        bboxes_list: List[BBoxes],
        labels_list: List[Labels],
    ) -> Tuple[np.ndarray, BBoxes, Labels]:
        """Apply mosaic augmentation to a group of 4 images.

        If the random draw exceeds mosaic_prob, the first image is returned
        after applying the standard pipeline instead.

        Parameters
        ----------
        images:
            List of exactly 4 images.
        bboxes_list, labels_list:
            Corresponding bbox and label lists for each image.

        Returns
        -------
        aug_image, aug_bboxes, aug_labels
        """
        if len(images) != 4:
            raise ValueError("apply_mosaic() requires exactly 4 images")

        if random.random() < self.mosaic_prob:
            mosaic_img, merged_bboxes, merged_labels = mosaic_4(
                images, bboxes_list, labels_list, self.image_size
            )
            # Apply non-geometric augmentations on the mosaic result
            mosaic_result = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.4),
                    A.HueSaturationValue(p=0.3),
                ],
                bbox_params=A.BboxParams(
                    format="pascal_voc",
                    label_fields=["class_ids"],
                    min_area=32 * 32,
                    min_visibility=0.3,
                ),
            )(image=mosaic_img, bboxes=merged_bboxes, class_ids=merged_labels)
            return (
                mosaic_result["image"],
                list(mosaic_result["bboxes"]),
                list(mosaic_result["class_ids"]),
            )
        else:
            return self.apply(images[0], bboxes_list[0], labels_list[0])
