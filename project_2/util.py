import tensorflow as tf
from tensorflow.keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt
import os

def read_data(validation_split=0.1):
    # Load CIFAR-100 dataset and normalize pixel values to [0, 1]
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Compute mean and std for normalization
    mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
    std = np.std(x_train, axis=(0, 1, 2), keepdims=True)

    # Normalize training and test data
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # Split a portion of training data for validation
    val_len = int(validation_split * len(x_train))
    x_val, y_val = x_train[:val_len], y_train[:val_len]
    x_train, y_train = x_train[val_len:], y_train[val_len:]
    # Normalize validation data
    x_val = (x_val - mean) / std

    return x_train, y_train, x_val, y_val, x_test, y_test

def plot_metrics(history, model_name):
    # Create results directory if it doesn't exist
    os.makedirs("project_2/results", exist_ok=True)

    # Plot training and validation accuracy over epochs
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'project_2/results/{model_name}_accuracy.png')
    plt.close()

    # Plot training and validation loss over epochs
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'project_2/results/{model_name}_loss.png')
    plt.close()

def compute_confidence_interval(acc, n, z=1.96):
    # Compute 95% confidence interval for accuracy
    se = np.sqrt((acc * (1 - acc)) / n)
    return acc - z * se, acc + z * se

def save_summary_to_file(model_name, test_acc, ci_low, ci_high, filename="project_2/model_summaries.txt"):
    # Append model results and confidence interval to summary file
    with open(filename, "a") as f:
        f.write(f"{model_name} Summary\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"95% Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}]\n")
        f.write("-" * 40 + "\n")