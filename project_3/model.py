from typing import Optional
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model


# ATTENTION 

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
            weights = weights * tf.expand_dims(tf.cast(mask, tf.float32), -1)
            weights = weights / (tf.reduce_sum(weights, axis=1, keepdims=True) + 1e-8)
        context = tf.reduce_sum(weights * hidden_states, axis=1)  # [B,H]
        return context, tf.squeeze(weights, axis=-1)


# BiLSTM++ 

def build_bilstm_attention_model(
    vocab_size: int,
    max_len: int,
    embed_dim: int = 200,
    lstm_units: int = 128,
    dropout_rate: float = 0.4,
    recurrent_dropout: float = 0.2,
    l2_reg: Optional[float] = 1e-4,
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

    x = layers.SpatialDropout1D(0.3)(x)

    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True,
                    dropout=dropout_rate, recurrent_dropout=recurrent_dropout),
        name="bilstm_1",
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True,
                    dropout=dropout_rate, recurrent_dropout=recurrent_dropout),
        name="bilstm_2",
    )(x)

    context, _ = BahdanauAttention(attn_units=lstm_units)(x, mask=mask)
    x = layers.Dropout(dropout_rate)(context)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg) if l2_reg else None)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="BiLSTM_Attention_Stacked")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    return model


# Transformer++ (2 blocks, positions)

def transformer_block(x, num_heads, d_model, ff_dim, dropout_rate, l2_reg):
    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model, name=None)(x, x)
    attn_out = layers.Dropout(dropout_rate)(attn_out)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)

    ffn = tf.keras.Sequential(
        [
            layers.Dense(ff_dim, activation="relu",
                         kernel_regularizer=regularizers.l2(l2_reg) if l2_reg else None),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model),
        ]
    )
    ffn_out = ffn(x)
    ffn_out = layers.Dropout(dropout_rate)(ffn_out)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_out)
    return x


def build_transformer_model(
    vocab_size: int,
    max_len: int,
    embed_dim: int = 192,        
    num_heads: int = 6,
    ff_dim: int = 768,
    dropout_rate: float = 0.3,
    l2_reg: Optional[float] = 1e-4,
) -> Model:
    inputs = layers.Input(shape=(max_len,), dtype="int32")

    tok_emb = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=True,
        embeddings_regularizer=regularizers.l2(l2_reg) if l2_reg else None,
        name="token_embed",
    )(inputs)

    # Learned positional embeddings
    pos_idx = tf.range(start=0, limit=max_len, delta=1)
    pos_emb_layer = layers.Embedding(input_dim=max_len, output_dim=embed_dim, name="pos_embed")
    pos_emb = pos_emb_layer(pos_idx)[None, ...]                      # [1, T, D]
    x = tok_emb + pos_emb[:, :tf.shape(tok_emb)[1], :]

    # Two encoder blocks
    x = transformer_block(x, num_heads, embed_dim, ff_dim, dropout_rate, l2_reg)
    x = transformer_block(x, num_heads, embed_dim, ff_dim, dropout_rate, l2_reg)

    # Pooling
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])

    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(l2_reg) if l2_reg else None)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="Transformer_Attention_Positional_2Blocks")
    try:
        opt = tf.keras.optimizers.experimental.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    except Exception:
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    return model