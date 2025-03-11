import numpy as np
import tensorflow as tf


class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def get_encoder() -> tf.keras.Model:
    latent_dim = 2
    encoder_inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu',
        strides=2,
        padding='same',
    )(encoder_inputs)
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        activation='relu',
        strides=2,
        padding='same',
    )(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=16, activation='relu')(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    return tf.keras.Model(
        encoder_inputs,
        [z_mean, z_log_var, z],
        name='encoder',
    )


def get_decoder() -> tf.keras.Model:
    latent_dim = 2
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
    x = tf.keras.layers.Reshape((7, 7, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(
        filters=64,
        kernel_size=3,
        activation='relu',
        strides=2,
        padding='same',
    )(x)
    x = tf.keras.layers.Conv2DTranspose(
        filters=32,
        kernel_size=3,
        activation='relu',
        strides=2,
        padding='same',
    )(x)
    decoder_outputs = tf.keras.layers.Conv2DTranspose(
        filters=1,
        kernel_size=3,
        activation='sigmoid',
        padding='same',
    )(x)
    return tf.keras.Model(
        latent_inputs,
        decoder_outputs,
        name='decoder',
    )


class VAE(tf.keras.Model):

    def __init__(
        self,
        encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
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

    def train_step(self, inputs: tf.Tensor):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(inputs)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(
                        inputs,
                        reconstruction,
                    ),
                    axis=(1, 2),
                ),
            )
            # noinspection PyTypeChecker
            kl_loss = -0.5 * (
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }


def train_vae() -> None:
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype('float32') / 255
    vae = VAE(
        encoder=get_encoder(),
        decoder=get_decoder(),
    )
    vae.compile(optimizer=tf.keras.optimizers.Adam())
    vae.fit(mnist_digits, epochs=30, batch_size=128)


if __name__ == '__main__':
    train_vae()
