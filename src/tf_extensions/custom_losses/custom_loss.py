from dataclasses import dataclass
from typing import Any, Union

import tensorflow as tf

from tf_extensions.custom_losses.adaptive_nmae import AdaptiveNMAE
from tf_extensions.custom_losses.base_loss import BaseLoss, BaseLossConfig
from tf_extensions.custom_losses.ms_dssim import MultiScaleDSSIM


@dataclass
class CustomLossConfig(BaseLossConfig):
    name: str = 'custom_loss'
    ssim_weight: float = 0.85

    def __post_init__(self) -> None:
        if self.ssim_weight < 0 or self.ssim_weight > 1:
            msg = 'SSIM weight {0} is out of range [0; 1].'.format(
                self.ssim_weight,
            )
            raise ValueError(msg)


class CustomLoss(BaseLoss):
    """Class for the CustomLoss."""
    config_type = CustomLossConfig

    def __init__(
        self,
        adaptive_nmae: Union[AdaptiveNMAE, dict[str, Any]] = None,
        ms_dssim: Union[MultiScaleDSSIM, dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.adaptive_nmae = self.get_loss_attribute(
            config=adaptive_nmae,
            loss_cls=AdaptiveNMAE,
        )
        self.ms_dssim = self.get_loss_attribute(
            config=ms_dssim,
            loss_cls=MultiScaleDSSIM,
        )

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        y_true, y_pred = self.cast_to_dtype(y_true=y_true, y_pred=y_pred)
        mae = self.adaptive_nmae(y_true, y_pred)
        ms_dssim = self.ms_dssim(y_true, y_pred)
        mae_weight = 1 - self.config.ssim_weight
        return mae_weight * mae + self.config.ssim_weight * ms_dssim
