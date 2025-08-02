
import tensorflow as tf
# handles data loading and evaluation
def build_model(architecture=1, use_regularizer=False):
    regularizer = tf.keras.regularizers.l2(0.001) if use_regularizer else None
    # Define a simple feedforward neural network architecture
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    # Add layers based on the selected architecture
    # Architecture 1: Two hidden layers
    # Architecture 2: Three hidden layers
    if architecture == 1:
        model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizer))
        model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizer))
    elif architecture == 2:
        model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizer))
        model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizer))
        model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizer))
    else:
        # If an invalid architecture is selected, raise an error
        raise ValueError("Invalid architecture selected. Choose 1 or 2.")
    # Output layer for classification
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model
 