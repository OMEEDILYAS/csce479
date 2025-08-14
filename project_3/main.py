import argparse
import json
import os
from typing import Dict, Any

import numpy as np
import tensorflow as tf

from util import (
    prepare_datasets,
    confusion_matrix,
    accuracy_from_cm,
    ci_95_for_accuracy,
    timestamped_dir,
    ensure_dir,
    save_history_plots,
    plot_confusion_matrix,
    write_text_summary,
)
from model import build_bilstm_attention_model, build_transformer_model


def train_and_eval(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    out_dir: str,
    tag: str,
    patience: int = 5,
    epochs: int = 20,
) -> Dict[str, Any]:
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
        )
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2)

    # Save history PNGs
    save_history_plots(history.history, os.path.join(out_dir, tag))

    # Evaluate and compute confusion matrix + CI on the test set
    y_true, y_pred = [], []
    for xb, yb in test_ds:
        probs = model.predict(xb, verbose=0).ravel()
        preds = (probs >= 0.5).astype(np.int32)
        y_pred.extend(preds.tolist())
        y_true.extend(yb.numpy().astype(np.int32).tolist())

    cm = confusion_matrix(y_true, y_pred, num_classes=2)
    acc = accuracy_from_cm(cm)
    num_correct = int(acc * len(y_true))
    ci_lo, ci_hi = ci_95_for_accuracy(num_correct, len(y_true))

    # Save CM PNG
    plot_confusion_matrix(cm.numpy(), os.path.join(out_dir, f"{tag}_cm.png"), normalize=True)

    # Evaluate built-in metrics
    test_metrics_list = model.evaluate(test_ds, verbose=0)
    # Keras returns [loss, metric1, metric2 ...]
    metrics_names = ["loss"] + [m.name if hasattr(m, "name") else f"metric_{i}" for i, m in enumerate(model.metrics, start=1)]
    test_metrics = {k: float(v) for k, v in zip(metrics_names, test_metrics_list)}

    out = {
        "history": history.history,
        "confusion_matrix": cm.numpy().tolist(),
        "accuracy": acc,
        "ci95": (ci_lo, ci_hi),
        "test_metrics": test_metrics,
    }
    # Save JSON summary too why not
    with open(os.path.join(out_dir, f"{tag}_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--run", type=str, default="all", choices=["bilstm", "transformer", "all"])
    parser.add_argument("--runs_dir", type=str, default="runs")  # where PNG/TXT/JSON go
    parser.add_argument("--save_models", action="store_true")
    args = parser.parse_args()

    # Repro-ish defaults
    tf.random.set_seed(42)
    np.random.seed(42)

    # Data (TFDS, with val split from training set)
    train_ds, val_ds, test_ds, vectorizer, sizes = prepare_datasets(
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
    )

    # Output directories
    root_out = timestamped_dir(args.runs_dir)

    results = {}

    # BiLSTM+Attention (two hyperparameter settings)
    if args.run in ("bilstm", "all"):
        bilstm_configs = [
            {"embed_dim": 128, "lstm_units": 128, "dropout_rate": 0.5, "l2_reg": 0.0},
            {"embed_dim": 256, "lstm_units": 256, "dropout_rate": 0.3, "l2_reg": 1e-4},
        ]
        for i, cfg in enumerate(bilstm_configs, start=1):
            tag = f"bilstm_{i}"
            tag_dir = ensure_dir(os.path.join(root_out, tag))
            model = build_bilstm_attention_model(
                vocab_size=args.vocab_size,
                max_len=args.max_len,
                **cfg,
            )
            print(f"\n=== Training {tag} with config: {cfg} ===")
            res = train_and_eval(model, train_ds, val_ds, test_ds, out_dir=tag_dir, tag=tag,
                                 patience=args.patience, epochs=args.epochs)
            results[tag] = {"config": cfg, **res}
            # TXT summary for quick inclusion in report
            write_text_summary(os.path.join(tag_dir, f"{tag}_summary.txt"), tag, cfg, res)
            # Optional: save model
            if args.save_models:
                model.save(os.path.join(tag_dir, f"{tag}_model.keras"))

    # Transformer (two hyperparameter settings)
    if args.run in ("transformer", "all"):
        transformer_configs = [
            {"embed_dim": 128, "num_heads": 4, "ff_dim": 256, "dropout_rate": 0.2, "l2_reg": 0.0},
            {"embed_dim": 256, "num_heads": 8, "ff_dim": 512, "dropout_rate": 0.3, "l2_reg": 1e-4},
        ]
        for i, cfg in enumerate(transformer_configs, start=1):
            tag = f"transformer_{i}"
            tag_dir = ensure_dir(os.path.join(root_out, tag))
            model = build_transformer_model(
                vocab_size=args.vocab_size,
                max_len=args.max_len,
                **cfg,
            )
            print(f"\n=== Training {tag} with config: {cfg} ===")
            res = train_and_eval(model, train_ds, val_ds, test_ds, out_dir=tag_dir, tag=tag,
                                 patience=args.patience, epochs=args.epochs)
            results[tag] = {"config": cfg, **res}
            write_text_summary(os.path.join(tag_dir, f"{tag}_summary.txt"), tag, cfg, res)
            if args.save_models:
                model.save(os.path.join(tag_dir, f"{tag}_model.keras"))

    # Overall index file
    with open(os.path.join(root_out, "all_results_index.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Console recap
    for name, info in results.items():
        cm = np.array(info["confusion_matrix"])
        print(f"\n{name} results")
        print(f"config: {info['config']}")
        print(f"test metrics: {info['test_metrics']}")
        print(f"accuracy: {info['accuracy']:.4f}  CI95: [{info['ci95'][0]:.4f}, {info['ci95'][1]:.4f}]")
        print("confusion_matrix:\n", cm)
    print(f"\nOutputs saved under: {root_out}")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()