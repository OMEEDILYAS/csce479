import os  # used to create folders and save files
# Import the TensorFlow Datasets library for loading datasets
import tensorflow_datasets as tfds
# Import TensorFlow for building and training models
import tensorflow as tf
# Import NumPy for numerical operations
import numpy as np
# Import confusion_matrix function for evaluation
from sklearn.metrics import confusion_matrix
# Import matplotlib for plotting
import matplotlib.pyplot as plt
# Import itertools for efficient looping
import itertools


def load_fashion_mnist(validation_split=0.2):
    # Load the Fashion MNIST dataset (train and test splits)
    ds_train_full, ds_test = tfds.load('fashion_mnist', split=['train', 'test'], as_supervised=True)

    # Shuffle the full training dataset for randomness
    ds_train_full = ds_train_full.shuffle(10000, seed=42)
    # Calculate the size of the validation set
    val_size = int(60000 * validation_split)
    # Take the first val_size samples for validation
    ds_val = ds_train_full.take(val_size)
    # Skip the validation samples to get the training set
    ds_train = ds_train_full.skip(val_size)

    # Function to normalize images and one-hot encode labels
    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255.0, tf.one_hot(label, depth=10)

    # Map normalization, batch, and prefetch for training set
    ds_train = ds_train.map(normalize_img).batch(128).prefetch(tf.data.AUTOTUNE)
    # Map normalization, batch, and prefetch for validation set
    ds_val = ds_val.map(normalize_img).batch(128).prefetch(tf.data.AUTOTUNE)
    # Map normalization, batch, and prefetch for test set
    ds_test = ds_test.map(normalize_img).batch(128).prefetch(tf.data.AUTOTUNE)

    # Return the prepared datasets
    return ds_train, ds_val, ds_test

def plot_confusion_matrix(cm, class_names,run_id=None):
    # Create a new figure for the confusion matrix
    plt.figure(figsize=(8, 6))
    # Display the confusion matrix as an image
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # Set the title of the plot
    plt.title('Confusion Matrix')
    # Add a color bar to the side
    plt.colorbar()
    # Set tick marks for the class names
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Format for the numbers in the matrix
    fmt = 'd'
    # Threshold for text color
    thresh = cm.max() / 2.
    # Add text annotations to each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # Label the axes
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Adjust layout for better fit
    plt.tight_layout()
    # Save confusion matrix image if run ID is provided
    if run_id is not None:
       os.makedirs("results", exist_ok=True)
       plt.savefig(f"results/conf_matrix_run_{run_id}.png")

    # Show the plot
    plt.show()

def evaluate_model(model, dataset, run_id=None):
    # Lists to store true and predicted labels
    y_true = []
    y_pred = []
    # Iterate over the dataset batches
    for images, labels in dataset:
        # Get model predictions for the batch
        preds = model.predict(images)
        # Append true labels (as integers)
        y_true.extend(tf.argmax(labels, axis=1).numpy())
        # Append predicted labels (as integers)
        y_pred.extend(tf.argmax(preds, axis=1).numpy())

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Plot the confusion matrix with class names
    plot_confusion_matrix(cm, class_names=["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                                           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"], run_id=run_id)

    # Calculate and print the accuracy
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"Accuracy: {acc:.4f}")
    # Save accuracy to results log if run ID is provided
    if 'run_id' in globals() and run_id is not None:
        os.makedirs("results", exist_ok=True)
        with open("results/accuracy_log.txt", "a") as f:
            f.write(f"Run {run_id}: Accuracy = {acc:.4f}\n")