#  model definitions (BiLSTM+attention, Transformer)
from typing import Optional
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from keras import ops as K  # Keras 3-safe ops for symbolic shapes/tensors


# ATTENTION LAYER 

class BahdanauAttention(layers.Layer):
    """
    Additive attention over time steps.
    - Consumes the time mask (so padding tokens don't get attention weight).
    - Stops mask propagation (so downstream loss doesn't receive a [B,T] mask).
    """
    def __init__(self, attn_units: int):
        super().__init__()
        self.W1 = layers.Dense(attn_units)  # project hidden states
        self.W2 = layers.Dense(attn_units)  # second projection (same input for simplicity)
        self.V = layers.Dense(1)            # score to a single logit per timestep
        self.supports_masking = True        # we accept a mask from previous layers

    def call(self, hidden_states, mask=None):
        # hidden_states: [B, T, H]; mask: [B, T] or None
        score = self.V(tf.nn.tanh(self.W1(hidden_states) + self.W2(hidden_states)))  # [B,T,1]
        weights = tf.nn.softmax(score, axis=1)  # attention weights over time [B,T,1]

        if mask is not None:
            # Mask out padding tokens and re-normalize the weights across time.
            mask = K.cast(mask, "float32")             # [B,T]
            weights = weights * K.expand_dims(mask, -1)
            denom = K.sum(weights, axis=1, keepdims=True) + 1e-8
            weights = weights / denom

        # Weighted sum across time -> a fixed-size context vector [B,H]
        context = K.sum(weights * hidden_states, axis=1)
        # Return context and (optionally useful) attention weights [B,T]
        return context, K.squeeze(weights, axis=-1)

    def compute_mask(self, inputs, mask=None):
        # We removed the time dimension (reduced over T), so do NOT propagate a time mask further.
        return None


# BiLSTM WITH ATTENTION 

def build_bilstm_attention_model(
    vocab_size: int,
    max_len: int,
    embed_dim: int = 200,
    lstm_units: int = 128,
    dropout_rate: float = 0.4,
    recurrent_dropout: float = 0.2,
    l2_reg: Optional[float] = 1e-4,
) -> Model:
    # Inputs are integer token IDs of fixed length
    inputs = layers.Input(shape=(max_len,), dtype="int32")

    # Token embeddings; mask_zero=True so Keras tracks padding positions
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=True,
        embeddings_regularizer=regularizers.l2(l2_reg) if l2_reg else None,
        name="embed",
    )(inputs)

    # Drop entire embedding channels (good regularizer for text)
    x = layers.SpatialDropout1D(0.3)(x)

    # First BiLSTM pass (keep sequence outputs for the next LSTM)
    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout
        ),
        name="bilstm_1",
    )(x)

    # Second BiLSTM pass (still returning sequence for attention)
    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout
        ),
        name="bilstm_2",
    )(x)

    # Attention over time (mask is passed automatically)
    context, _ = BahdanauAttention(attn_units=lstm_units)(x)

    # Small classification head
    x = layers.Dropout(dropout_rate)(context)
    x = layers.Dense(
        128, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg) if l2_reg else None
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    # Binary output (positive vs negative)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="BiLSTM_Attention_Stacked")
    # Compile with Adam and standard binary loss/metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # fallback to Adam
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    return model


# TRANSFORMER BUILDING BLOCK

def transformer_block(x, num_heads, d_model, ff_dim, dropout_rate, l2_reg):
    # Multi-head self-attention + residual + layernorm
    attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn_out = layers.Dropout(dropout_rate)(attn_out)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)

    # Feed-forward network + residual + layernorm
    ffn = tf.keras.Sequential(
        [
            layers.Dense(
                ff_dim, activation="relu",
                kernel_regularizer=regularizers.l2(l2_reg) if l2_reg else None
            ),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model),
        ]
    )
    ffn_out = ffn(x)
    ffn_out = layers.Dropout(dropout_rate)(ffn_out)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_out)
    return x


# TRANSFORMER MODEL

def build_transformer_model(
    vocab_size: int,
    max_len: int,
    embed_dim: int = 192,       # d_model size
    num_heads: int = 6,         # number of attention heads
    ff_dim: int = 768,          # inner FFN dimension
    dropout_rate: float = 0.3,
    l2_reg: Optional[float] = 1e-4,
) -> Model:
    # Token ID input
    inputs = layers.Input(shape=(max_len,), dtype="int32")

    # Token embeddings with mask
    tok_emb = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=True,
        embeddings_regularizer=regularizers.l2(l2_reg) if l2_reg else None,
        name="token_embed",
    )(inputs)

    # Learned positional embeddings (use Keras ops for shapes so it's Keras 3–safe)
    pos_idx = K.arange(max_len)                                 # [T]
    pos_emb_layer = layers.Embedding(input_dim=max_len, output_dim=embed_dim, name="pos_embed")
    pos_emb = pos_emb_layer(pos_idx)[None, ...]                 # [1, T, D]
    seq_len = K.shape(tok_emb)[1]                               # dynamic sequence length
    x = tok_emb + pos_emb[:, :seq_len, :]                       # add (broadcast) positions to tokens

    # Two Transformer encoder blocks
    x = transformer_block(x, num_heads, embed_dim, ff_dim, dropout_rate, l2_reg)
    x = transformer_block(x, num_heads, embed_dim, ff_dim, dropout_rate, l2_reg)

    # Pooling: combine average and max pooled features across time
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])

    # Classification head
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(
        128, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg) if l2_reg else None
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="Transformer_Attention_Positional_2Blocks")

    # Try AdamW (weight decay) if available; otherwise use Adam
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