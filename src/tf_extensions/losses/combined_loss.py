from dataclasses import dataclass
from typing import Any, TypeVar

import tensorflow as tf

from tf_extensions.losses.base_loss import BaseLoss, BaseLossConfig
from tf_extensions.losses.supported_losses import supported_losses


@dataclass
class CombinedLossConfig(BaseLossConfig):
    name: str = 'combined_loss'
    losses: list = None
    weights: list = None


CombinedLossInstance = TypeVar('CombinedLossInstance', bound='CombinedLoss')


class CombinedLoss(BaseLoss):
    """
    A flexible loss class that combines multiple loss functions with weights.

    Parameters
    ----------
    losses : list
        Initialized loss objects.
    weights : list
        Corresponding weights for each loss.

    """
    config_type = CombinedLossConfig

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.config.losses is None or self.config.weights is None:
            msg = 'Losses and weights must be provided as lists.'
            raise ValueError(msg)
        if len(self.config.losses) != len(self.config.weights):
            msg = 'Losses and weights lists must have the same length.'
            raise ValueError(msg)
        name = kwargs.get('name', '')
        if name:
            self.config.name = name
        else:
            self.config.name = '_'.join(
                [loss.name for loss in self.config.losses],
            )
        self.name = self.config.name

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """Compute the combined loss."""
        y_true, y_pred = self.cast_to_dtype(y_true=y_true, y_pred=y_pred)
        weighted_losses = [
            weight * loss_fn(y_true, y_pred)
            for loss_fn, weight in zip(self.config.losses, self.config.weights)
        ]
        return tf.cast(sum(weighted_losses), dtype=self.config.dtype)

    def get_config(self) -> dict[str, Any]:  # noqa: WPS615
        config = super().get_config()
        loss_configs = []
        for loss in self.config.losses:
            loss_configs.append(loss.get_config())
        config['losses'] = loss_configs
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CombinedLossInstance:
        combined_loss_config = {}
        for attr_name, attr_value in config.items():
            if attr_name == 'losses':
                losses = []
                for loss_config in attr_value:
                    losses.append(
                        supported_losses[
                            loss_config['cls_name']
                        ].from_config(loss_config),
                    )
                combined_loss_config[attr_name] = losses
            else:
                combined_loss_config[attr_name] = attr_value
        return cls(**combined_loss_config)
