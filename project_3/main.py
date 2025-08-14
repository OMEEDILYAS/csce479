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
    collect_probs_and_labels,
    tune_threshold_on_val,
)
from model import build_bilstm_attention_model, build_transformer_model


def train_and_eval(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    out_dir: str,
    tag: str,
    patience: int = 6,
    epochs: int = 20,
    threshold_metric: str = "accuracy",
) -> Dict[str, Any]:
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=patience, restore_best_weights=True),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2)

    # Save history PNGs
    save_history_plots(history.history, os.path.join(out_dir, tag))

    # Threshold tuning on validation
    val_probs, val_y = collect_probs_and_labels(val_ds, model)
    best_t = tune_threshold_on_val(val_probs, val_y, metric=threshold_metric, sweep=(0.3, 0.7, 41))

    # Evaluate on test with tuned threshold
    test_probs, test_y = collect_probs_and_labels(test_ds, model)
    test_preds = (test_probs >= best_t).astype(int)
    cm = confusion_matrix(test_y, test_preds, num_classes=2)
    acc = accuracy_from_cm(cm)
    num_correct = int(acc * len(test_y))
    ci_lo, ci_hi = ci_95_for_accuracy(num_correct, len(test_y))

    # Save CM PNG
    plot_confusion_matrix(cm, os.path.join(out_dir, f"{tag}_cm.png"), normalize=True)

    # Also record Keras metrics at default threshold for reference
    test_metrics_list = model.evaluate(test_ds, verbose=0)
    metrics_names = ["loss"] + [m.name if hasattr(m, "name") else f"metric_{i}" for i, m in enumerate(model.metrics, start=1)]
    test_metrics = {k: float(v) for k, v in zip(metrics_names, test_metrics_list)}

    out = {
        "history": history.history,
        "threshold": float(best_t),
        "confusion_matrix": cm.tolist(),
        "accuracy": acc,
        "ci95": (ci_lo, ci_hi),
        "test_metrics": test_metrics,  # Keras metrics at default threshold (0.5)
    }
    # Save JSON + TXT summaries
    with open(os.path.join(out_dir, f"{tag}_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    write_text_summary(os.path.join(out_dir, f"{tag}_summary.txt"), tag, model.get_config() if hasattr(model, "get_config") else {"name": model.name}, out)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=30000)  # bumped to 30k (fewer OOVs)
    parser.add_argument("--max_len", type=int, default=320)       # bumped to 320 (more context)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--run", type=str, default="all", choices=["bilstm", "transformer", "all"])
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--threshold_metric", type=str, default="accuracy", choices=["accuracy", "f1"])
    args = parser.parse_args()

    # Repro-ish defaults
    tf.random.set_seed(42)
    np.random.seed(42)

    # Data
    train_ds, val_ds, test_ds, _, _ = prepare_datasets(
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
    )

    # Output directories
    root_out = timestamped_dir(args.runs_dir)
    results = {}

    if args.run in ("bilstm", "all"):
        bilstm_configs = [
            # Regularized stacked BiLSTM
            {"embed_dim": 200, "lstm_units": 128, "dropout_rate": 0.4, "recurrent_dropout": 0.2, "l2_reg": 1e-4},
            {"embed_dim": 256, "lstm_units": 192, "dropout_rate": 0.4, "recurrent_dropout": 0.2, "l2_reg": 1e-4},
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
                                 patience=args.patience, epochs=args.epochs, threshold_metric=args.threshold_metric)
            results[tag] = {"config": cfg, **res}
            if args.save_models:
                model.save(os.path.join(tag_dir, f"{tag}_model.keras"))

    if args.run in ("transformer", "all"):
        transformer_configs = [
            {"embed_dim": 192, "num_heads": 6, "ff_dim": 768, "dropout_rate": 0.3, "l2_reg": 1e-4},
            {"embed_dim": 256, "num_heads": 8, "ff_dim": 1024, "dropout_rate": 0.3, "l2_reg": 1e-4},
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
                                 patience=args.patience, epochs=args.epochs, threshold_metric=args.threshold_metric)
            results[tag] = {"config": cfg, **res}
            if args.save_models:
                model.save(os.path.join(tag_dir, f"{tag}_model.keras"))

    with open(os.path.join(root_out, "all_results_index.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Console recap
    for name, info in results.items():
        cm = np.array(info["confusion_matrix"])
        print(f"\n{name} results")
        print(f"config: {info['config']}")
        print(f"threshold: {info['threshold']:.3f}")
        print(f"test metrics (Keras @0.5): {info['test_metrics']}")
        print(f"accuracy (tuned): {info['accuracy']:.4f}  CI95: [{info['ci95'][0]:.4f}, {info['ci95'][1]:.4f}]")
        print("confusion_matrix:\n", cm)
    print(f"\nOutputs saved under: {root_out}")


if __name__ == "__main__":
    tf.get_logger().setLevel("ERROR")
    main()