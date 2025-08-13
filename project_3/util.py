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

AUTOTUNE = tf.data.AUTOTUNE


#DATA 

def load_imdb_tfds() -> Tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
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
    vectorizer = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=max_len,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
    )
    vectorizer.adapt(train_text_ds)
    return vectorizer


def vectorize_dataset(
    ds: tf.data.Dataset,
    vectorizer: TextVectorization,
    batch_size: int,
    cache: bool = True,
) -> tf.data.Dataset:
    def _vec_map(x, y):
        x = vectorizer(x)
        return x, y
    ds = ds.map(_vec_map, num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def prepare_datasets(
    vocab_size: int,
    max_len: int,
    batch_size: int,
    val_fraction: float = 0.2,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, TextVectorization, Dict[str, int]]:
    raw_train, raw_test, info = load_imdb_tfds()
    train_split, val_split = split_train_val(raw_train, val_fraction=val_fraction)
    train_text_only = train_split.map(lambda x, y: x)
    vectorizer = make_vectorizer(train_text_only, vocab_size=vocab_size, max_len=max_len)
    train_ds = vectorize_dataset(train_split, vectorizer, batch_size=batch_size)
    val_ds = vectorize_dataset(val_split, vectorizer, batch_size=batch_size)
    test_ds = vectorize_dataset(raw_test, vectorizer, batch_size=batch_size)
    sizes = {"train": 25000 - int(25000 * val_fraction), "val": int(25000 * val_fraction), "test": 25000}
    return train_ds, val_ds, test_ds, vectorizer, sizes


# METRICS 

def confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int = 2) -> tf.Tensor:
    y_true = tf.convert_to_tensor(y_true, dtype=tf.int32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.int32)
    return tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes)


def accuracy_from_cm(cm: tf.Tensor) -> float:
    cm = tf.cast(cm, tf.float32)
    correct = tf.linalg.trace(cm)
    total = tf.reduce_sum(cm)
    return float(correct / (total + 1e-8))


def ci_95_for_accuracy(num_correct: int, n: int) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = num_correct / n
    se = math.sqrt(p * (1 - p) / n)
    lo = max(0.0, p - 1.96 * se)
    hi = min(1.0, p + 1.96 * se)
    return (lo, hi)


# SAVING / PLOTTING 

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def timestamped_dir(base: str = "runs") -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    return ensure_dir(os.path.join(base, ts))


def save_history_plots(history: Dict[str, List[float]], out_png_prefix: str) -> None:
    # Accuracy
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

    # Loss
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
            plt.text(j, i, format(val, fmt),
                     horizontalalignment="center",
                     color="white" if cm_to_plot[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=160)
    plt.close()


def write_text_summary(path: str, name: str, config: Dict[str, Any], results: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Model: {name}\n")
        f.write(f"Config: {json.dumps(config)}\n")
        f.write(f"Test metrics: {results['test_metrics']}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}  CI95: [{results['ci95'][0]:.4f}, {results['ci95'][1]:.4f}]\n")
        f.write("Confusion matrix (rows=true, cols=pred):\n")
        for row in results["confusion_matrix"]:
            f.write(f"{row}\n")
