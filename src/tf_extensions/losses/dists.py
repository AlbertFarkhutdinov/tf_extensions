from dataclasses import dataclass

import tensorflow as tf

from tf_extensions.losses.ssim_calculator import SSIMCalculator
from tf_extensions.losses.vgg_base import VGGBase, VGGBaseConfig


@dataclass
class DISTSConfig(VGGBaseConfig):
    name: str = 'dists'
    texture_weight: float = 0.5
    texture_stability_constant: float = 1e-6
    structure_stability_constant: float = 1e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.texture_weight < 0 or self.texture_weight > 1:
            msg = 'Texture weight {0} is out of range [0; 1].'.format(
                self.texture_weight,
            )
            raise ValueError(msg)


class DISTS(VGGBase):
    """Class for the DISTS."""

    config_type = DISTSConfig

    def _get_features(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tuple[list[tf.Tensor], list[tf.Tensor]]:
        true_features, pred_features = super()._get_features(
            y_true=y_true,
            y_pred=y_pred,
        )
        true_features = [y_true, *true_features]
        pred_features = [y_pred, *pred_features]
        return true_features, pred_features

    def _compute_loss(
        self,
        true_features: list[tf.Tensor],
        pred_features: list[tf.Tensor],
    ) -> tf.Tensor:
        structure_loss = 0
        texture_loss = 0

        feat_number = len(true_features)
        for true_feat, pred_feat in zip(true_features, pred_features):
            texture, structure = SSIMCalculator(
                tensor1=tf.cast(true_feat, self.config.dtype),
                tensor2=tf.cast(pred_feat, self.config.dtype),
                c1=self.config.texture_stability_constant,
                c2=self.config.structure_stability_constant,
                averaging='reduce_mean',
                averaging_kwargs={'axis': (1, 2)},
            ).calculate()
            texture_loss += tf.reduce_mean(texture, axis=-1) / feat_number
            structure_loss += tf.reduce_mean(structure, axis=-1) / feat_number

        texture_loss *= self.config.texture_weight
        structure_loss *= (1 - self.config.texture_weight)
        return 1 - (structure_loss + texture_loss)
