import tensorflow as tf
from tensorflow.keras import layers, regularizers

# Model A: Simpler CNN with two convolutional blocks and one dense layer
def build_model_a(input_shape=(32, 32, 3), num_classes=100, dropout_rate=0.5):
    model = tf.keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001), input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        # Second convolutional block
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        # Dense layers for classification
        layers.Flatten(),
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')  # Output layer
    ])
    return model

# Model B: Deeper CNN with more filters and dense units
def build_model_b(input_shape=(32, 32, 3), num_classes=100, dropout_rate=0.5):
    model = tf.keras.Sequential([
        # First convolutional block with two Conv2D layers
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001), input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        # Second convolutional block with two Conv2D layers
        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        # Dense layers for classification
        layers.Flatten(),
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax') # Output layer
    ])
    return model