"""Module providing a class for VAE losses."""
from dataclasses import dataclass

import tensorflow as tf

from tf_extensions.losses.base_loss import BaseLoss, BaseLossConfig


def get_kl_loss(z_mean: tf.Tensor, z_log_var: tf.Tensor) -> tf.Tensor:
    """
    Return the Kullback-Leibler (KL) divergence loss for the latent variables.

    Parameters
    ----------
    z_mean : tf.Tensor
        The mean of the latent variable distribution.
    z_log_var : tf.Tensor
        The log variance of the latent variable distribution.

    Returns
    -------
    tf.Tensor
        The computed KL loss.

    """
    # noinspection PyTypeChecker
    z_mean_square = tf.square(z_mean)
    z_var = tf.exp(z_log_var)
    return tf.reduce_mean(
        tf.reduce_sum(
            0.5 * (z_var + z_mean_square - z_log_var - 1),
            axis=1,
        ),
    )


@dataclass
class VAEReconstructionLossConfig(BaseLossConfig):
    """
    Configuration class for VAE reconstruction loss.

    Attributes
    ----------
    name : str
        Name of the loss function, default is 'vae_reconstruction'.

    """

    name: str = 'vae_reconstruction'


class VAEReconstructionLoss(BaseLoss):
    """
    Class implementing reconstruction loss for variational autoencoders (VAE).

    Attributes
    ----------
    config : VAEReconstructionLossConfig
        Configuration of VAEReconstructionLoss.

    """

    config_type = VAEReconstructionLossConfig

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """
        Return the reconstruction loss between the true and predicted images.

        Parameters
        ----------
        y_true : array-like
            Ground truth images.
        y_pred : array-like
            Predicted images.

        Returns
        -------
        float
            The computed reconstruction loss.

        """
        return tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(y_true, y_pred),
                axis=(1, 2),
            ),
        )
