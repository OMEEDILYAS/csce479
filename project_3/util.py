# helper functions for data, metrics, plotting, and saving
import os
import math
import json
import time
from typing import Tuple, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import TextVectorization

# Let tf.data tune parallelism / prefetch automatically
AUTOTUNE = tf.data.AUTOTUNE


# DATA LOADING / PREP 

def load_imdb_tfds() -> Tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
    # Load IMDB from TensorFlow Datasets.
    # Returns supervised pairs (text, label) and dataset metadata.
    (train_ds, test_ds), info = tfds.load(
        "imdb_reviews",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
    )
    return train_ds, test_ds, info


def split_train_val(
    train_ds: tf.data.Dataset,
    val_fraction: float = 0.2,
    shuffle_buffer: int = 25000,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    # Deterministically split the 25k training examples into train/val.
    # We shuffle once with a fixed seed, then take a slice for validation.
    train_ds = train_ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=False)
    total = 25000
    val_count = int(total * val_fraction)
    val_ds = train_ds.take(val_count)
    new_train_ds = train_ds.skip(val_count)
    return new_train_ds, val_ds


def make_vectorizer(
    train_text_ds: tf.data.Dataset,
    vocab_size: int,
    max_len: int,
) -> TextVectorization:
    # Build a TextVectorization layer and learn (adapt) the vocabulary on *training text only*
    # to avoid leaking validation/test information.
    vectorizer = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_len,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
    )
    vectorizer.adapt(train_text_ds)  # learn top tokens from train set only
    return vectorizer


def vectorize_dataset(
    ds: tf.data.Dataset,
    vectorizer: TextVectorization,
    batch_size: int,
    cache: bool = True,
) -> tf.data.Dataset:
    # Convert raw (text, label) pairs into (token_id_tensor, label) and build a fast pipeline.

    def _vec_map(x, y):
        x = vectorizer(x)  # text -> padded token ids
        return x, y

    ds = ds.map(_vec_map, num_parallel_calls=AUTOTUNE)  # parallel vectorization
    if cache:
        ds = ds.cache()  # keep in memory if it fits (faster subsequent epochs)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)  # efficient input pipeline
    return ds


def prepare_datasets(
    vocab_size: int,
    max_len: int,
    batch_size: int,
    val_fraction: float = 0.2,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, TextVectorization, Dict[str, int]]:
    # End-to-end data prep: load TFDS, split train/val, build vectorizer on train text,
    # and return vectorized train/val/test + dataset sizes (for reference).
    raw_train, raw_test, _ = load_imdb_tfds()
    train_split, val_split = split_train_val(raw_train, val_fraction=val_fraction)

    # Build a text-only dataset for adapting the vectorizer (no labels).
    train_text_only = train_split.map(lambda x, y: x)
    vectorizer = make_vectorizer(train_text_only, vocab_size=vocab_size, max_len=max_len)

    # Vectorize each split.
    train_ds = vectorize_dataset(train_split, vectorizer, batch_size=batch_size)
    val_ds = vectorize_dataset(val_split, vectorizer, batch_size=batch_size)
    test_ds = vectorize_dataset(raw_test, vectorizer, batch_size=batch_size)

    sizes = {
        "train": 25000 - int(25000 * val_fraction),
        "val": int(25000 * val_fraction),
        "test": 25000,
    }
    return train_ds, val_ds, test_ds, vectorizer, sizes


# METRICS 

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 2) -> np.ndarray:
    # Build a confusion matrix (rows=true label, cols=predicted label).
    y_true = tf.convert_to_tensor(y_true, dtype=tf.int32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.int32)
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes)
    return cm.numpy()


def accuracy_from_cm(cm: np.ndarray) -> float:
    # Accuracy is (sum of diagonal) / (sum of all cells).
    correct = np.trace(cm.astype(np.float64))
    total = cm.sum()
    return float(correct / (total + 1e-8))


def ci_95_for_accuracy(num_correct: int, n: int) -> Tuple[float, float]:
    # 95% CI for accuracy using the normal approximation.
    if n <= 0:
        return (0.0, 1.0)
    p = num_correct / n
    se = math.sqrt(p * (1 - p) / n)
    lo = max(0.0, p - 1.96 * se)
    hi = min(1.0, p + 1.96 * se)
    return (lo, hi)


# SAVING / PLOTTING 

def ensure_dir(path: str) -> str:
    # Make sure a directory exists and return its path (convenience).
    os.makedirs(path, exist_ok=True)
    return path


def timestamped_dir(base: str = "runs") -> str:
    # Create a new runs/<YYYYmmdd-HHMMSS> folder and return it.
    ts = time.strftime("%Y%m%d-%H%M%S")
    return ensure_dir(os.path.join(base, ts))


def save_history_plots(history: Dict[str, List[float]], out_png_prefix: str) -> None:
    # Save training/validation accuracy and loss curves as PNGs using a filename prefix.
    os.makedirs(os.path.dirname(out_png_prefix), exist_ok=True)

    # Accuracy curve
    plt.figure()
    if "accuracy" in history:
        plt.plot(history["accuracy"], label="train_acc")
    if "val_accuracy" in history:
        plt.plot(history["val_accuracy"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(f"{out_png_prefix}_accuracy.png", dpi=160)
    plt.close()

    # Loss curve
    plt.figure()
    if "loss" in history:
        plt.plot(history["loss"], label="train_loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss")
    plt.tight_layout()
    plt.savefig(f"{out_png_prefix}_loss.png", dpi=160)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, out_png_path: str, normalize: bool = True, labels: List[str] = None) -> None:
    # Save a confusion matrix heatmap; normalize rows for readability by default.
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)

    plt.figure()
    if labels is None:
        labels = ["neg", "pos"]
    cm_to_plot = cm.astype("float")
    if normalize:
        row_sums = cm_to_plot.sum(axis=1, keepdims=True) + 1e-8
        cm_to_plot = cm_to_plot / row_sums
    plt.imshow(cm_to_plot, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    fmt = ".2f" if normalize else "d"
    thresh = cm_to_plot.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_to_plot[i, j] if normalize else cm[i, j]
            plt.text(
                j, i, format(val, fmt),
                horizontalalignment="center",
                color="white" if cm_to_plot[i, j] > thresh else "black",
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=160)
    plt.close()


def write_text_summary(path: str, name: str, config: Dict[str, Any], results: Dict[str, Any]) -> None:
    # Save a human-readable TXT summary (model config, metrics, and the CM numbers).
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Model: {name}\n")
        f.write(f"Config: {json.dumps(config)}\n")
        f.write(f"Test metrics: {results['test_metrics']}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}  CI95: [{results['ci95'][0]:.4f}, {results['ci95'][1]:.4f}]\n")
        f.write("Confusion matrix (rows=true, cols=pred):\n")
        for row in results["confusion_matrix"]:
            f.write(f"{row}\n")


# THRESHOLD TUNING 

def collect_probs_and_labels(ds: tf.data.Dataset, model: tf.keras.Model) -> Tuple[np.ndarray, np.ndarray]:
    # Run inference on a dataset split and return probabilities + labels as numpy arrays.
    probs, labels = [], []
    for xb, yb in ds:
        p = model.predict(xb, verbose=0).ravel()     # predicted probabilities (sigmoid output)
        probs.append(p)
        labels.append(yb.numpy().astype(int))        # corresponding true labels
    return np.concatenate(probs), np.concatenate(labels)


def tune_threshold_on_val(
    val_probs: np.ndarray,
    val_y: np.ndarray,
    metric: str = "accuracy",
    sweep: Tuple[float, float, int] = (0.3, 0.7, 41)
) -> float:
    # Sweep decision thresholds on the validation set and pick the best one.
    ts = np.linspace(*sweep)  # e.g., 0.30, 0.31, ..., 0.70
    if metric == "accuracy":
        scores = [(t, ((val_probs >= t).astype(int) == val_y).mean()) for t in ts]
    else:
        # F1 option: compute precision/recall & F1 at each threshold
        scores = []
        for t in ts:
            preds = (val_probs >= t).astype(int)
            tp = ((preds == 1) & (val_y == 1)).sum()
            fp = ((preds == 1) & (val_y == 0)).sum()
            fn = ((preds == 0) & (val_y == 1)).sum()
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            scores.append((t, f1))
    best_t = max(scores, key=lambda x: x[1])[0]  # choose threshold with the best score
    return float(best_t)
