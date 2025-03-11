from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from tf_extensions.losses.base_loss import BaseLoss, BaseLossConfig
from tf_extensions.losses.ssim_calculator import SSIMCalculator


@dataclass
class SSIMBaseConfig(BaseLossConfig):
    max_val: int = 2
    filter_size: int = 5
    filter_sigma: float = 1.5
    k1: float = 0.01
    k2: float = 0.03


@dataclass
class DSSIMConfig(SSIMBaseConfig):
    name: str = 'dssim'
    return_cs_map: bool = False
    return_index_map: bool = False
    with_channels_averaging: bool = True


class DSSIM(BaseLoss):
    """Class for the DSSIM."""

    config_type = DSSIMConfig

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
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
        y_true, y_pred = self.cast_to_dtype(y_true=y_true, y_pred=y_pred)
        # max_val - depth of image
        # (255 in case the image has a different scale)
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
        """Function to mimic the 'fspecial' gaussian MATLAB function."""
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
        tensor = tf.expand_dims(tensor, axis=-1)
        tensor = tf.repeat(tensor, n_channel, axis=-1)
        tensor = tf.expand_dims(tensor, axis=-1)
        tensor = tf.repeat(tensor, repeats=1, axis=-1)
        return tf.cast(tensor, dtype=self.config.dtype)

    def _get_conv_kwargs(self, n_channel: int) -> dict[str, Any]:
        window = self._tf_fspecial_gauss(n_channel=n_channel)
        window = tf.cast(window, self.config.dtype)
        return {
            'filter': window,
            'strides': [1, 1, 1, 1],
            'padding': 'VALID',
        }
