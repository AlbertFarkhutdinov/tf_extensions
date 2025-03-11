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


def get_dataset() -> tuple[
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    """Return rescale and reshaped MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / x_train.max()
    x_test = x_test.astype(np.float32) / x_test.max()
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)


def get_encoder(
    input_shape: tuple[int, ...],
    intermediate_dimension: int,
    latent_dimension: int,
) -> tf.keras.Model:
    encoder_inputs = tf.keras.Input(shape=input_shape)
    intermediate = tf.keras.layers.Flatten()(encoder_inputs)
    intermediate = tf.keras.layers.Dense(
        intermediate_dimension,
        activation='relu',
    )(intermediate)
    z_mean = tf.keras.layers.Dense(
        latent_dimension,
        name='z_mean',
    )(intermediate)
    z_log_var = tf.keras.layers.Dense(
        latent_dimension,
        name='z_log_var',
    )(intermediate)
    z = Sampling()([z_mean, z_log_var])
    return tf.keras.Model(
        encoder_inputs,
        [z_mean, z_log_var, z],
        name='encoder',
    )


def get_decoder(
    input_shape: tuple[int, ...],
    intermediate_dimension: int,
    latent_dimension: int,
) -> tf.keras.Model:
    latent_inputs = tf.keras.Input(shape=(latent_dimension,))
    decoder_outputs = tf.keras.layers.Dense(
        intermediate_dimension,
        activation='relu',
    )(latent_inputs)
    decoder_outputs = tf.keras.layers.Dense(
        np.array(input_shape).prod(),
        activation='sigmoid',
    )(decoder_outputs)
    decoder_outputs = tf.keras.layers.Reshape(input_shape)(decoder_outputs)
    return tf.keras.Model(
        latent_inputs,
        decoder_outputs,
        name='decoder',
    )


def get_kl_loss(z_mean: tf.Tensor, z_log_var: tf.Tensor) -> tf.Tensor:
    # noinspection PyTypeChecker
    return tf.reduce_mean(
        tf.reduce_sum(
            0.5 * (tf.exp(z_log_var) + tf.square(z_mean) - 1 - z_log_var),
            axis=1,
        ),
    )


def get_reconstruction_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
) -> tf.Tensor:
    return tf.reduce_mean(
        tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(y_true, y_pred),
            axis=(1, 2),
        ),
    )


class VariationalAutoEncoder(tf.keras.Model):

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
            reconstruction_loss = get_reconstruction_loss(
                y_true=inputs,
                y_pred=reconstruction,
            )
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
