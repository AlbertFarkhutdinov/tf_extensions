import keras
import numpy as np
import tensorflow as tf


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs, *args, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def get_dataset():
    """Return rescale and reshaped MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / x_train.max()
    x_test = x_test.astype(np.float32) / x_test.max()
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)


def get_encoder(
    input_shape,
    intermediate_dimension: int,
    latent_dimension: int,
):
    encoder_inputs = keras.Input(shape=input_shape)
    intermediate = keras.layers.Flatten()(encoder_inputs)
    intermediate = keras.layers.Dense(
        intermediate_dimension,
        activation='relu',
    )(intermediate)
    z_mean = keras.layers.Dense(
        latent_dimension,
        name='z_mean',
    )(intermediate)
    z_log_var = keras.layers.Dense(
        latent_dimension,
        name='z_log_var',
    )(intermediate)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model(
        encoder_inputs,
        [z_mean, z_log_var, z],
        name='encoder',
    )


def get_decoder(
    input_shape,
    intermediate_dimension: int,
    latent_dimension: int,
):
    latent_inputs = keras.Input(shape=(latent_dimension,))
    decoder_outputs = keras.layers.Dense(
        intermediate_dimension,
        activation='relu',
    )(latent_inputs)
    decoder_outputs = keras.layers.Dense(
        np.array(input_shape).prod(),
        activation='sigmoid',
    )(decoder_outputs)
    decoder_outputs = keras.layers.Reshape(input_shape)(decoder_outputs)
    return keras.Model(
        latent_inputs,
        decoder_outputs,
        name='decoder',
    )


def get_kl_loss(z_mean, z_log_var):
    return tf.reduce_mean(
        tf.reduce_sum(
            0.5 * (tf.exp(z_log_var) + tf.square(z_mean) - 1 - z_log_var),
            axis=1,
        ),
    )


def get_reconstruction_loss(data, reconstruction):
    return tf.reduce_mean(
        tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(data, reconstruction),
            axis=(1, 2),
        ),
    )


class VariationalAutoEncoder(tf.keras.Model):

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name='reconstruction_loss',
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name='kl_loss')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = get_reconstruction_loss(data, reconstruction)
            kl_loss = get_kl_loss(z_mean, z_log_var)
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {metric.name: metric.result() for metric in self.metrics}


def train_vae() -> None:
    (x_train, y_train), (x_test, _) = get_dataset()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    kwargs = {
        'input_shape': mnist_digits.shape[1:],
        'intermediate_dimension': 32,
        'latent_dimension': 2,
    }
    encoder = get_encoder(**kwargs)
    encoder.summary()
    decoder = get_decoder(**kwargs)
    decoder.summary()
    vae = VariationalAutoEncoder(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam())
    vae.fit(mnist_digits, epochs=30, batch_size=128)


if __name__ == '__main__':
    train_vae()
