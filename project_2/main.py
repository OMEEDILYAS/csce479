import tensorflow as tf
from model import build_model_a, build_model_b
from util import read_data, plot_metrics, compute_confidence_interval, save_summary_to_file


open("project_2/model_summaries.txt", "w").close()

x_train, y_train, x_val, y_val, x_test, y_test = read_data()
 
models = {
    "Model_A": build_model_a(),
    "Model_B": build_model_b()
}

for model_name, model in models.items():
    print(f"\n Training {model_name}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=15,
        restore_best_weights=True,
        monitor="val_accuracy"
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.3,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=64,
        validation_data=(x_val, y_val),
        callbacks=[early_stop, reduce_lr],
        verbose=2
    )

    # Plot metrics
    plot_metrics(history, model_name)

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    ci_low, ci_high = compute_confidence_interval(test_acc, len(x_test))

    print(f"\n {model_name} Test Accuracy: {test_acc:.4f} | 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    save_summary_to_file(model_name, test_acc, ci_low, ci_high)
