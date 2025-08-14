from typing import Optional
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model


class BahdanauAttention(layers.Layer):
    def __init__(self, attn_units: int):
        super().__init__()
        self.W1 = layers.Dense(attn_units)
        self.W2 = layers.Dense(attn_units)
        self.V = layers.Dense(1)

    def call(self, hidden_states, mask=None):
        score = self.V(tf.nn.tanh(self.W1(hidden_states) + self.W2(hidden_states)))  # [B,T,1]
        weights = tf.nn.softmax(score, axis=1)  # [B,T,1]
        if mask is not None:
            mask = tf.cast(mask, tf.float32)  # [B,T]
            weights = weights * tf.expand_dims(mask, -1)
            weights = weights / (tf.reduce_sum(weights, axis=1, keepdims=True) + 1e-8)
        context = tf.reduce_sum(weights * hidden_states, axis=1)  # [B,H]
        return context, tf.squeeze(weights, axis=-1)


def build_bilstm_attention_model(
    vocab_size: int,
    max_len: int,
    embed_dim: int = 128,
    lstm_units: int = 128,
    dropout_rate: float = 0.5,
    l2_reg: Optional[float] = 0.0,
) -> Model:
    inputs = layers.Input(shape=(max_len,), dtype="int32")
    mask = tf.cast(tf.not_equal(inputs, 0), tf.bool)

    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=True,
        embeddings_regularizer=regularizers.l2(l2_reg) if l2_reg else None,
        name="embed",
    )(inputs)

    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate),
        name="bilstm",
    )(x)

    context, _ = BahdanauAttention(attn_units=lstm_units)(x, mask=mask)
    x = layers.Dropout(dropout_rate)(context)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg) if l2_reg else None)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="BiLSTM_Attention")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_transformer_model(
    vocab_size: int,
    max_len: int,
    embed_dim: int = 128,
    num_heads: int = 4,
    ff_dim: int = 256,
    dropout_rate: float = 0.2,
    l2_reg: Optional[float] = 0.0,
) -> Model:
    inputs = layers.Input(shape=(max_len,), dtype="int32")

    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=True,
        embeddings_regularizer=regularizers.l2(l2_reg) if l2_reg else None,
        name="embed",
    )(inputs)

    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, name="mha")(x, x)
    attn_out = layers.Dropout(dropout_rate)(attn_out)
    x1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)

    ffn = tf.keras.Sequential(
        [
            layers.Dense(ff_dim, activation="relu",
                         kernel_regularizer=regularizers.l2(l2_reg) if l2_reg else None),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
        ],
        name="ffn",
    )
    ffn_out = ffn(x1)
    ffn_out = layers.Dropout(dropout_rate)(ffn_out)
    x2 = layers.LayerNormalization(epsilon=1e-6)(x1 + ffn_out)

    x2 = layers.GlobalAveragePooling1D()(x2)
    x2 = layers.Dropout(dropout_rate)(x2)
    x2 = layers.Dense(128, activation="relu",
                      kernel_regularizer=regularizers.l2(l2_reg) if l2_reg else None)(x2)
    x2 = layers.Dropout(dropout_rate)(x2)
    outputs = layers.Dense(1, activation="sigmoid")(x2)

    model = Model(inputs, outputs, name="Transformer_Attention")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model