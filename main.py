import os  # for creating result folders
# Import TensorFlow and our helper functions from util and model files
import tensorflow as tf
# handles data loading and evaluation
from util import load_fashion_mnist, evaluate_model
# defines model architectures
from model import build_model  

# Load the Fashion-MNIST dataset and split it into training, validation, and test sets
ds_train, ds_val, ds_test = load_fashion_mnist()
# Make sure folders exist for saving results and models
os.makedirs("results", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# Define the different architectures, learning rates, and regularization options we want to test
architectures = [1, 2]  # two different model setups
hyperparams = [0.001, 0.0005]  # trying two learning rates
regularizers = [False, True]  # with and without L2 regularization

run = 1  # keep track of how many total runs we've done
for arch in architectures:
    for lr in hyperparams:
        for reg in regularizers:
            # Show the current settings for this run
            print(f"\nRun {run}: Architecture={arch}, LearningRate={lr}, Regularizer={reg}")

            # Build the model using the selected architecture and regularizer setting
            model = build_model(architecture=arch, use_regularizer=reg)

            # Compile the model with the chosen learning rate and standard classification setup
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),  # set optimizer with current learning rate
                loss='categorical_crossentropy',  # good loss choice for softmax + one-hot labels
                metrics=['accuracy']  # track accuracy during training
            )

            # Train the model on the training set and validate on the validation set
            model.fit(ds_train, validation_data=ds_val, epochs=10, verbose=2)

            # After training, evaluate the model using the test data
            print("Evaluating on test set:")
            evaluate_model(model, ds_test,run_id=run)
            # Save detailed run info to a summary log file
            with open("results/run_summaries.txt", "a") as f:
                f.write(f"Run {run}\n")
                f.write(f"Architecture: {arch}\n")
                f.write(f"Learning rate: {lr}\n")
                f.write(f"Regularizer: {reg}\n")
                f.write(f"Accuracy: {model.evaluate(ds_test, verbose=0)[1]:.4f}\n")
                f.write("-" * 28 + "\n")
            # Save the model to disk
            tf.saved_model.save(model, f"saved_models/run_{run}")


            run += 1  # update the run number for the next configuration
