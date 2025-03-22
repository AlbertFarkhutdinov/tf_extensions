import numpy as np
import tensorflow as tf

from tf_extensions.layers.vae_sampling import VAESampling
from tf_extensions.losses.vae_loss import VAEReconstructionLoss, get_kl_loss


def get_encoder(
    intermediate_dimension: int = 32,
    input_shape: tuple[int, ...] = (28, 28, 1),
    latent_dimension: int = 2,
    with_conv: bool = True,
) -> tf.keras.Model:
    encoder_inputs = tf.keras.Input(shape=input_shape)
    intermediate = encoder_inputs

    if with_conv:
        intermediate = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            activation='relu',
            strides=2,
            padding='same',
        )(intermediate)
        intermediate = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation='relu',
            strides=2,
            padding='same',
        )(intermediate)

    intermediate = tf.keras.layers.Flatten()(intermediate)
    intermediate = tf.keras.layers.Dense(
        units=intermediate_dimension,
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
    z = VAESampling()([z_mean, z_log_var])
    return tf.keras.Model(
        encoder_inputs,
        [z_mean, z_log_var, z],
        name='encoder',
    )


def get_decoder(
    intermediate_dimension: int = None,
    input_shape: tuple[int, ...] = (7, 7, 64),
    latent_dimension: int = 2,
    with_conv: bool = True,
) -> tf.keras.Model:
    latent_inputs = tf.keras.Input(
        shape=(latent_dimension,),
    )
    decoder_outputs = latent_inputs
    if intermediate_dimension:
        decoder_outputs = tf.keras.layers.Dense(
            intermediate_dimension,
            activation='relu',
        )(decoder_outputs)
    decoder_outputs = tf.keras.layers.Dense(
        np.array(input_shape).prod(),
        activation='relu',
    )(decoder_outputs)
    decoder_outputs = tf.keras.layers.Reshape(
        target_shape=input_shape,
    )(decoder_outputs)

    if with_conv:
        decoder_outputs = tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=3,
            activation='relu',
            strides=2,
            padding='same',
        )(decoder_outputs)
        decoder_outputs = tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=3,
            activation='relu',
            strides=2,
            padding='same',
        )(decoder_outputs)
        decoder_outputs = tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=3,
            activation='sigmoid',
            padding='same',
        )(decoder_outputs)

    return tf.keras.Model(
        latent_inputs,
        decoder_outputs,
        name='decoder',
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
        self.total_loss_tracker = tf.keras.metrics.Mean(
            name='total_loss',
        )
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name='reconstruction_loss',
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(
            name='kl_loss',
        )

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
            reconstruction_loss = VAEReconstructionLoss()(
                y_true=inputs,
                y_pred=reconstruction,
            )
            kl_loss = get_kl_loss(
                z_mean=z_mean,
                z_log_var=z_log_var,
            )
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {metric.name: metric.result() for metric in self.metrics}


def train_vae() -> None:
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    mnist_digits = np.concatenate(
        [x_train, x_test],
        axis=0,
    )
    mnist_digits = np.expand_dims(mnist_digits, -1).astype('float32') / 255
    kwargs = {
        'input_shape': mnist_digits.shape[1:],
        'intermediate_dimension': 32,
        'latent_dimension': 2,
    }
    encoder = get_encoder(**kwargs)
    encoder.summary()
    decoder = get_decoder(**kwargs)
    decoder.summary()
    vae = VariationalAutoEncoder(
        encoder=encoder,
        decoder=decoder,
    )
    vae.compile(optimizer=tf.keras.optimizers.Adam())
    vae.fit(mnist_digits, epochs=30, batch_size=128)


if __name__ == '__main__':
    train_vae()
