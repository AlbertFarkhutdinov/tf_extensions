"""The module provides sampling layer for variational autoencoder (VAE)."""
import tensorflow as tf

from tf_extensions.layers.base_layer import BaseLayer


class VAESampling(BaseLayer):
    """
    Sampling layer for variational autoencoder (VAE).

    This layer uses (z_mean, z_log_var) to sample the latent vector z.

    """

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        Return the sampled latent vector using the reparameterization trick.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, height, width, channels).

        Returns
        -------
        tf.Tensor
            Output tensor after applying the ASPP transformation.

        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
