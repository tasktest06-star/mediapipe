"""train_model.py — Train an EfficientDet-Lite2 object-detector for 150 custom
classes using mediapipe_model_maker.

Usage
-----
python train_model.py \\
    --config configs/training_config.yaml \\
    --output_dir /models/efficientdet_lite2_150cls
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import yaml
import tensorflow as tf
from mediapipe_model_maker import object_detector


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate the YAML training configuration.

    Parameters
    ----------
    config_path:
        Path to training_config.yaml.

    Returns
    -------
    config:
        Nested dictionary of configuration values.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    required_sections = ["model", "dataset", "training", "output", "export"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: [{section}]")
    return config


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def build_dataset(tfrecord_pattern: str, class_map_path: str,
                  is_training: bool) -> object_detector.Dataset:
    """Build a MediaPipe Model Maker Dataset from TFRecord shards.

    Parameters
    ----------
    tfrecord_pattern:
        Glob pattern for TFRecord files, e.g. '/data/train/shard-*.tfrecord'.
    class_map_path:
        Path to class_map.json produced by prepare_dataset.py.
    is_training:
        Whether this dataset is for training (shuffled) or evaluation.

    Returns
    -------
    dataset:
        mediapipe_model_maker Dataset object.
    """
    with open(class_map_path) as f:
        raw_map = json.load(f)
    # Model Maker expects a list of label names sorted by index
    label_map = {int(k): v for k, v in raw_map.items()}
    label_names = [label_map[i] for i in sorted(label_map.keys())]

    dataset = object_detector.Dataset.from_tfrecord(
        tfrecord_pattern=tfrecord_pattern,
        label_map=label_map,
        num_shards=None,       # auto-detect
        shuffle=is_training,
    )
    return dataset


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

def build_model_spec(config: Dict[str, Any]) -> object_detector.SupportedModels:
    """Select the EfficientDet-Lite variant from config.

    Parameters
    ----------
    config:
        Full config dict from load_config().

    Returns
    -------
    model_spec:
        MediaPipe Model Maker model specification enum value.
    """
    arch = config["model"]["architecture"].lower()
    mapping = {
        "efficientdet_lite0": object_detector.SupportedModels.EFFICIENTDET_LITE0,
        "efficientdet_lite1": object_detector.SupportedModels.EFFICIENTDET_LITE1,
        "efficientdet_lite2": object_detector.SupportedModels.EFFICIENTDET_LITE2,
        "efficientdet_lite3": object_detector.SupportedModels.EFFICIENTDET_LITE3,
    }
    if arch not in mapping:
        raise ValueError(f"Unsupported architecture: {arch}. Choose from {list(mapping)}")
    return mapping[arch]


def build_hparams(config: Dict[str, Any]) -> object_detector.HParams:
    """Construct HParams from the training section of the config.

    Parameters
    ----------
    config:
        Full config dict.

    Returns
    -------
    hparams:
        mediapipe_model_maker HParams instance.
    """
    t = config["training"]
    hparams = object_detector.HParams(
        learning_rate=t.get("learning_rate", 0.08),
        batch_size=t.get("batch_size", 16),
        epochs=t.get("epochs", 100),
        cosine_decay_epochs=t.get("epochs", 100) if t.get("lr_schedule") == "cosine" else None,
        warmup_epochs=t.get("lr_warmup_epochs", 5),
        export_dir=config["output"]["checkpoint_dir"],
    )
    return hparams


# ---------------------------------------------------------------------------
# Early stopping callback
# ---------------------------------------------------------------------------

class EarlyStoppingCallback(tf.keras.callbacks.EarlyStopping):
    """Thin wrapper that prints a message when training stops early."""

    def on_train_end(self, logs: Dict | None = None) -> None:
        super().on_train_end(logs)
        if self.stopped_epoch > 0:
            print(f"\nEarly stopping triggered at epoch {self.stopped_epoch + 1}.")
            print(f"Best weights restored from epoch {self.best_epoch + 1}.")


# ---------------------------------------------------------------------------
# Training loop with per-epoch logging
# ---------------------------------------------------------------------------

def train(config: Dict[str, Any], output_dir: str) -> None:
    """Run the full training pipeline.

    Parameters
    ----------
    config:
        Loaded configuration dictionary.
    output_dir:
        Directory where checkpoints, TFLite model, and logs are saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config["output"]["tensorboard_log_dir"], exist_ok=True)

    dataset_cfg = config["dataset"]
    class_map_path = dataset_cfg["class_map_path"]

    print("Loading training dataset...")
    train_data = build_dataset(dataset_cfg["train_tfrecord_pattern"], class_map_path, is_training=True)
    print(f"  Training samples: {len(train_data)}")

    print("Loading validation dataset...")
    val_data = build_dataset(dataset_cfg["val_tfrecord_pattern"], class_map_path, is_training=False)
    print(f"  Validation samples: {len(val_data)}")

    model_spec = build_model_spec(config)
    hparams = build_hparams(config)

    print(f"\nInitialising {config['model']['architecture']} for {config['model']['num_classes']} classes...")
    model = object_detector.ObjectDetector.create(
        train_data=train_data,
        model_spec=model_spec,
        validation_data=val_data,
        hparams=hparams,
    )

    # Build callbacks
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=config["output"]["tensorboard_log_dir"],
            histogram_freq=0,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "checkpoint_epoch_{epoch:03d}"),
            save_best_only=False,
            save_weights_only=True,
            verbose=0,
            period=config["output"].get("save_freq", 5),
        ),
    ]

    es_cfg = config["training"].get("early_stopping", {})
    if es_cfg.get("enabled", True):
        callbacks.append(
            EarlyStoppingCallback(
                monitor=es_cfg.get("monitor", "val_AP"),
                patience=es_cfg.get("patience", 15),
                min_delta=es_cfg.get("min_delta", 0.001),
                restore_best_weights=es_cfg.get("restore_best_weights", True),
                verbose=1,
            )
        )

    print("\nStarting training...\n")
    start_time = time.time()
    model.model.fit(
        train_data.gen_tf_dataset(
            model_spec,
            batch_size=config["training"]["batch_size"],
            is_training=True,
        ),
        epochs=config["training"]["epochs"],
        validation_data=val_data.gen_tf_dataset(
            model_spec,
            batch_size=config["training"]["batch_size"],
            is_training=False,
        ),
        callbacks=callbacks,
    )
    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed / 3600:.1f} hours.")

    # Export TFLite
    tflite_path = os.path.join(output_dir, config["export"]["tflite_filename"])
    print(f"\nExporting TFLite model to {tflite_path} ...")
    model.export_model(model_name=config["export"]["tflite_filename"], export_dir=output_dir)

    if config["output"].get("export_saved_model", False):
        saved_model_dir = os.path.join(output_dir, "saved_model")
        print(f"Exporting SavedModel to {saved_model_dir} ...")
        model.model.save(saved_model_dir)

    print(f"\nAll outputs saved to {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train EfficientDet-Lite2 with MediaPipe Model Maker.")
    parser.add_argument("--config", required=True, help="Path to training_config.yaml")
    parser.add_argument("--output_dir", required=True, help="Directory for model outputs")
    parser.add_argument("--epochs", type=int, help="Override epochs from config")
    parser.add_argument("--batch_size", type=int, help="Override batch size from config")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    # Log GPU availability
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPUs detected: {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if config["training"].get("mixed_precision", True):
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision (float16/float32) enabled.")
    else:
        print("WARNING: No GPU detected. Training will be very slow on CPU.")

    train(config, args.output_dir)


if __name__ == "__main__":
    main()
