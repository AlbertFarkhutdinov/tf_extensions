from typing import Optional, Union

import numpy as np
import tensorflow as tf


class MultiHeadSelfAttention(tf.keras.layers.Layer):

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.depth = embed_dim // num_heads

        self.wq = tf.keras.layers.Dense(embed_dim)
        self.wk = tf.keras.layers.Dense(embed_dim)
        self.wv = tf.keras.layers.Dense(embed_dim)

        self.dense = tf.keras.layers.Dense(embed_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, *args, **kwargs):
        batch_size = tf.shape(inputs)[0]

        q = self.split_heads(self.wq(inputs), batch_size)
        k = self.split_heads(self.wk(inputs), batch_size)
        v = self.split_heads(self.wv(inputs), batch_size)

        attn_weights = (
            tf.matmul(q, k, transpose_b=True)
            / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        )
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        attn_output = tf.matmul(attn_weights, v)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        concat_attn = tf.reshape(
            attn_output,
            shape=(batch_size, -1, self.embed_dim),
        )

        return self.dense(concat_attn)


class FeedForwardNetwork(tf.keras.layers.Layer):

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc2 = tf.keras.layers.Dense(embed_dim)

    def call(self, inputs, *args, **kwargs):
        return self.fc2(self.fc1(inputs))


class TransformerBlock(tf.keras.Model):

    def __init__(self, embed_dim, num_heads, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = FeedForwardNetwork(embed_dim, hidden_dim)

        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(
        self,
        inputs: tf.Tensor,
        training: Union[bool, tf.Tensor] = None,
        mask: Union[Optional[tf.Tensor], list[Optional[tf.Tensor]]] = None,
    ) -> tf.Tensor:
        attn_output = self.attn(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layer_norm2(out1 + ffn_output)


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, max_len, embed_dim):
        super().__init__()
        self.pos_encoding = self.positional_encoding(max_len, embed_dim)

    @classmethod
    def positional_encoding(cls, max_len, embed_dim):
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(embed_dim)[np.newaxis, :]
        angles = pos / np.power(10000, (2 * (i // 2)) / embed_dim)
        pos_encoding = np.zeros((max_len, embed_dim))
        pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])
        return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs, *args, **kwargs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class Transformer(tf.keras.Model):

    def __init__(
        self,
        vocab_size,
        max_len,
        embed_dim,
        num_heads,
        hidden_dim,
        num_layers,
    ):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(max_len, embed_dim)
        self.enc_layers = [
            TransformerBlock(embed_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(
        self,
        inputs: tf.Tensor,
        training: Union[bool, tf.Tensor] = None,
        mask: Union[Optional[tf.Tensor], list[Optional[tf.Tensor]]] = None,
    ) -> tf.Tensor:
        x = self.embedding(inputs)
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        for layer in self.enc_layers:
            x = layer(x, training)

        return self.final_layer(x)


def run():
    vocab_size = 10000
    max_len = 100
    embed_dim = 512
    num_heads = 8
    hidden_dim = 2048
    num_layers = 6

    transformer = Transformer(
        vocab_size=vocab_size,
        max_len=max_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    inputs = tf.random.uniform(
        shape=(2, max_len),
        dtype=tf.int32,
        minval=0,
        maxval=vocab_size,
    )
    outputs = transformer(inputs)
    return outputs.shape  # Expected: (batch_size, max_len, vocab_size)


if __name__ == '__main__':
    run()
