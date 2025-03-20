"""Module implementing the DSSIM (Structural Dissimilarity) loss function."""
from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from tf_extensions.losses.base_loss import BaseLoss, BaseLossConfig
from tf_extensions.losses.ssim_calculator import SSIMCalculator


@dataclass
class SSIMBaseConfig(BaseLossConfig):
    """
    Base configuration class for SSIM-based losses.

    Attributes
    ----------
    max_val : int, default: 2
        The dynamic range of the images
        (i.e., the difference between the max the and min allowed values).
    filter_size : int, default: 5
        Size of the Gaussian filter used for SSIM computation.
    filter_sigma : float, default: 1.5
        Standard deviation for the Gaussian filter.
    k1 : float, default: 0.01
        Constant stabilizing luminance term of the SSIM.
    k2 : float, default: 0.03
        Constant stabilizing contrast term of the SSIM.

    """

    max_val: int = 2
    filter_size: int = 5
    filter_sigma: float = 1.5
    k1: float = 0.01
    k2: float = 0.03


@dataclass
class DSSIMConfig(SSIMBaseConfig):
    """
    Configuration class for DSSIM loss.

    Attributes
    ----------
    name : str, default: "dssim"
        Name of the loss function.
    return_cs_map : bool, default: False
        Whether to return the contrast-structure component separately.
    return_index_map : bool, default: False
        Whether to return the full SSIM index map instead of a single scalar.
    with_channels_averaging : bool, default: True
        Whether to average over all channels.

    """

    name: str = 'dssim'
    return_cs_map: bool = False
    return_index_map: bool = False
    with_channels_averaging: bool = True


class DSSIM(BaseLoss):
    """
    Class implementing the DSSIM (Structural Dissimilarity) loss.

    DSSIM = (1 - SSIM) / 2

    Attributes
    ----------
    config : DSSIMConfig
        Configuration of DSSIM.

    """

    config_type = DSSIMConfig

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """
        Return the DSSIM loss between ground truth and predicted images.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth image tensor.
        y_pred : tf.Tensor
            Predicted image tensor.

        Returns
        -------
        tf.Tensor
            DSSIM loss value.

        """
        ssim = self.get_ssim(y_true=y_true, y_pred=y_pred)
        return tf.divide(
            tf.convert_to_tensor(value=1, dtype=self.config.dtype) - ssim,
            y=2,
        )

    def get_ssim(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """
        Return the SSIM index between ground truth and predicted images.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth image tensor.
        y_pred : tf.Tensor
            Predicted image tensor.

        Returns
        -------
        tf.Tensor
            SSIM index.

        """
        ssim_list = [
            self._ssim_per_channel(
                y_true[:, :, :, channel:channel + 1],
                y_pred[:, :, :, channel:channel + 1],
            )
            for channel in range(y_true.shape[-1])
        ]
        stacked_ssim = tf.stack(ssim_list, axis=-1)
        if self.config.with_channels_averaging:
            return tf.reduce_mean(stacked_ssim, axis=-1)
        return stacked_ssim

    def _ssim_per_channel(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """
        Return SSIM for each channel separately.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth image tensor.
        y_pred : tf.Tensor
            Predicted image tensor.

        Returns
        -------
        tf.Tensor
            SSIM value per channel.

        """
        y_true, y_pred = self.cast_to_dtype(y_true=y_true, y_pred=y_pred)
        luminance, contrast_struct = SSIMCalculator(
            tensor1=y_true,
            tensor2=y_pred,
            c1=(self.config.k1 * self.config.max_val) ** 2,
            c2=(self.config.k2 * self.config.max_val) ** 2,
            averaging='conv2d',
            averaging_kwargs=self._get_conv_kwargs(n_channel=y_true.shape[-1]),
        ).calculate()
        ssim = luminance * contrast_struct
        if self.config.return_cs_map:
            ssim = tf.stack(values=[ssim, contrast_struct], axis=0)
        if self.config.return_index_map:
            return tf.reduce_mean(ssim, axis=(-1))
        return tf.reduce_mean(ssim, axis=(-3, -2, -1))

    def _tf_fspecial_gauss(
        self,
        n_channel: int = 1,
    ) -> tf.Tensor:
        """
        Generate a Gaussian filter.

        Parameters
        ----------
        n_channel : int
            Number of channels for the filter.

        Returns
        -------
        tf.Tensor
            Gaussian filter tensor.

        """
        x_data, y_data = self._get_xy_data(n_channel=n_channel)
        sigma = self.config.filter_sigma
        arg_square = tf.divide(
            tf.pow(x_data, y=2) + tf.pow(y_data, y=2),
            sigma ** 2,
        )
        gauss = tf.exp(-arg_square / 2)
        return gauss / tf.reduce_sum(gauss)

    def _get_xy_data(
        self,
        n_channel: int = 1,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Generate meshgrid data for Gaussian filter.

        Parameters
        ----------
        n_channel : int
            Number of channels for the filter.

        Returns
        -------
        tuple of tf.Tensor
            Meshgrid tensors for x and y coordinates.

        """
        size = self.config.filter_size
        symmetric_range = tf.range(size) - size // 2
        x_data, y_data = tf.meshgrid(symmetric_range, symmetric_range)
        x_data = self._preprocess_for_fspecial(x_data, n_channel=n_channel)
        y_data = self._preprocess_for_fspecial(y_data, n_channel=n_channel)
        return x_data, y_data

    def _preprocess_for_fspecial(
        self,
        tensor: tf.Tensor,
        n_channel: int = 1,
    ) -> tf.Tensor:
        """
        Preprocess tensor for Gaussian filter creation.

        Parameters
        ----------
        tensor : tf.Tensor
            Meshgrid tensor for x or y coordinates.
        n_channel : int
            Number of channels for the filter.

        Returns
        -------
        tf.Tensor
            Processed tensor for Gaussian filter creation.

        """
        tensor = tf.expand_dims(tensor, axis=-1)
        tensor = tf.repeat(tensor, n_channel, axis=-1)
        tensor = tf.expand_dims(tensor, axis=-1)
        tensor = tf.repeat(tensor, repeats=1, axis=-1)
        return tf.cast(tensor, dtype=self.config.dtype)

    def _get_conv_kwargs(self, n_channel: int) -> dict[str, Any]:
        """
        Return convolution parameters for SSIM computation.

        Parameters
        ----------
        n_channel : int
            Number of channels for the filter.

        Returns
        -------
        dict
            Convolution parameters for SSIM computation.

        """
        window = self._tf_fspecial_gauss(n_channel=n_channel)
        window = tf.cast(window, self.config.dtype)
        return {
            'filter': window,
            'strides': [1, 1, 1, 1],
            'padding': 'VALID',
        }
