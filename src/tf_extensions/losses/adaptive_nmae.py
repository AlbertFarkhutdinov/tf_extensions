from dataclasses import dataclass

import tensorflow as tf
from keras import backend as kb

from tf_extensions.losses.base_loss import BaseLoss, BaseLossConfig


@dataclass
class AdaptiveNMAEConfig(BaseLossConfig):
    name: str = 'adaptive_nmae'
    scale: float = 13.5


class AdaptiveNMAE(BaseLoss):
    """Class for the Adaptive NMAE."""

    config_type = AdaptiveNMAEConfig

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        y_true, y_pred = self.cast_to_dtype(y_true=y_true, y_pred=y_pred)
        loss = kb.mean(
            tf.abs(y_pred - y_true),
            axis=[1, 2, 3],
        )
        mean_abs_true = kb.mean(
            tf.abs(y_true),
            axis=[1, 2, 3],
        )
        norm_coefficient = kb.switch(
            mean_abs_true != 0,
            mean_abs_true * self.config.scale,
            kb.ones_like(loss),
        )
        return tf.cast(loss / norm_coefficient, dtype=self.config.dtype)
