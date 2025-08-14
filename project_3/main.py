# train/evaluate pipeline
import argparse
import json
import os
from typing import Dict, Any

import numpy as np
import tensorflow as tf

from util import (
    prepare_datasets,          # data pipeline: load/split/vectorize
    confusion_matrix,          # build confusion matrix
    accuracy_from_cm,          # get accuracy from CM
    ci_95_for_accuracy,        # 95% CI for accuracy
    timestamped_dir,           # runs/<timestamp> helper
    ensure_dir,                # mkdir -p helper
    save_history_plots,        # save training curves
    plot_confusion_matrix,     # save confusion matrix heatmap
    write_text_summary,        # save a readable TXT summary
    collect_probs_and_labels,  # gather probs + labels from a dataset
    tune_threshold_on_val,     # choose decision threshold on validation
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
    """
    Train a model, tune its decision threshold on the validation split, evaluate on test,
    and save plots + JSON/TXT summaries to out_dir.
    """
    # Make sure the output directory exists
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Basic, effective callbacks: reduce LR on plateau + early stopping
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=patience, restore_best_weights=True),
    ]

    # Train the model (history records per-epoch metrics)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2)

    # Save accuracy/loss curves for the report
    save_history_plots(history.history, os.path.join(out_dir, tag))

    # Use validation set to tune the decision threshold (e.g., not always 0.5)
    val_probs, val_y = collect_probs_and_labels(val_ds, model)
    best_t = tune_threshold_on_val(val_probs, val_y, metric=threshold_metric, sweep=(0.3, 0.7, 41))

    # Final evaluation on the test set with the tuned threshold
    test_probs, test_y = collect_probs_and_labels(test_ds, model)
    test_preds = (test_probs >= best_t).astype(int)
    cm = confusion_matrix(test_y, test_preds, num_classes=2)
    acc = accuracy_from_cm(cm)
    num_correct = int(acc * len(test_y))
    ci_lo, ci_hi = ci_95_for_accuracy(num_correct, len(test_y))

    # Save a normalized confusion matrix image
    os.makedirs(out_dir, exist_ok=True)
    plot_confusion_matrix(cm, os.path.join(out_dir, f"{tag}_cm.png"), normalize=True)

    # Also record Keras' own evaluation at default threshold (0.5) for reference
    test_metrics_list = model.evaluate(test_ds, verbose=0)
    metrics_names = ["loss"] + [m.name if hasattr(m, "name") else f"metric_{i}" for i, m in enumerate(model.metrics, start=1)]
    test_metrics = {k: float(v) for k, v in zip(metrics_names, test_metrics_list)}

    # Bundle results
    out = {
        "history": history.history,
        "threshold": float(best_t),
        "confusion_matrix": cm.tolist(),
        "accuracy": acc,
        "ci95": (ci_lo, ci_hi),
        "test_metrics": test_metrics,
    }

    # Save JSON (machine-readable) and TXT (human-readable) summaries
    with open(os.path.join(out_dir, f"{tag}_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    write_text_summary(os.path.join(out_dir, f"{tag}_summary.txt"), tag, model.get_config() if hasattr(model, "get_config") else {"name": model.name}, out)

    return out


def main():
    # CLI arguments with sensible defaults for IMDB text classification
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=30000)  # larger vocab -> fewer OOVs
    parser.add_argument("--max_len", type=int, default=320)       # more context per review
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--run", type=str, default="all", choices=["bilstm", "transformer", "all"])
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--threshold_metric", type=str, default="accuracy", choices=["accuracy", "f1"])
    args = parser.parse_args()

    # Make results roughly reproducible across runs
    tf.random.set_seed(42)
    np.random.seed(42)

    # Build data pipelines (train/val/test) and the vectorizer
    train_ds, val_ds, test_ds, _, _ = prepare_datasets(
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
    )

    # Create a timestamped root folder to store all run artifacts
    root_out = os.path.abspath(timestamped_dir(args.runs_dir))
    results = {}

    #  BiLSTM experiments (two configurations)
    if args.run in ("bilstm", "all"):
        bilstm_configs = [
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

    #  Transformer experiments (two configurations) 
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

    # Save an index with all run results under this timestamp
    with open(os.path.join(root_out, "all_results_index.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print a quick console recap of each run
    for name, info in results.items():
        cm = np.array(info["confusion_matrix"])
        print(f"\n{name} results")
        print(f"config: {info['config']}")
        print(f"test metrics: {info['test_metrics']}")
        print(f"accuracy: {info['accuracy']:.4f}  CI95: [{info['ci95'][0]:.4f}, {info['ci95'][1]:.4f}]")
        print("confusion_matrix:\n", cm)
    print(f"\nOutputs saved under: {root_out}")


if __name__ == "__main__":
    # Quiet down TF logs 
    tf.get_logger().setLevel("ERROR")
    main()
