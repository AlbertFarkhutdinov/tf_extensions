from dataclasses import dataclass
from typing import Any, TypeVar

import tensorflow as tf

from tf_extensions.custom_losses.base_loss import BaseLoss, BaseLossConfig
from tf_extensions.custom_losses.supported_losses import supported_losses


@dataclass
class MultiScaleLossConfig(BaseLossConfig):
    name: str = 'multiscale_loss'
    base_loss: tf.keras.losses.Loss = None
    weights: list = None


MultiScaleLossInstance = TypeVar(
    'MultiScaleLossInstance',
    bound='MultiScaleLoss',
)


class MultiScaleLoss(BaseLoss):
    """
    A flexible loss class that combines multiple loss functions with weights.

    Parameters
    ----------
    losses : list
        Initialized loss objects.
    weights : list
        Corresponding weights for each loss.

    """
    config_type = MultiScaleLossConfig

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.config.base_loss is None:
            msg = 'Loss must be provided.'
            raise ValueError(msg)
        name = kwargs.get('name', '')
        if name:
            self.config.name = name
        else:
            self.config.name = 'multiscale_{0}'.format(
                self.config.base_loss.name,
            )
        self.name = self.config.name

    def call(
        self,
        y_true: tuple[tf.Tensor],
        y_pred: tuple[tf.Tensor],
    ) -> tf.Tensor:
        if not isinstance(y_true, tuple) or not isinstance(y_pred, tuple):
            raise ValueError('Inputs must be tuples of tensors.')
        if len(y_true) != len(y_pred):
            raise ValueError('Lengths of y_true and y_pred must match.')
        if self.config.weights and (len(self.config.weights) != len(y_true)):
            raise ValueError('Lengths of weights and y_true must match.')
        batch_size = y_true[0].shape[0]
        losses = []
        for level, _ in enumerate(y_true):
            self._check_batch_size(
                y_true=y_true[level],
                y_pred=y_pred[level],
                batch_size=batch_size,
            )
            loss = self.config.base_loss(y_true[level], y_pred[level])
            if self.config.weights:
                loss *= self.config.weights[level]
            losses.append(loss)
        return tf.cast(
            tf.reduce_mean(losses, axis=0),
            dtype=self.config.dtype,
        )

    def get_config(self) -> dict[str, Any]:  # noqa: WPS615
        config = super().get_config()
        config['base_loss'] = self.config.base_loss.get_config()
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MultiScaleLossInstance:
        combined_loss_config = {}
        for attr_name, attr_value in config.items():
            if attr_name == 'base_loss':
                combined_loss_config[attr_name] = supported_losses[
                    attr_value['cls_name']
                ].from_config(attr_value)
            else:
                combined_loss_config[attr_name] = attr_value
        return cls(**combined_loss_config)

    @classmethod
    def _check_batch_size(
        cls,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        batch_size: int,
    ) -> None:
        if y_true.shape[0] != batch_size:
            raise ValueError('Batch sizes in y_true must match.')
        if y_pred.shape[0] != batch_size:
            raise ValueError('Batch sizes in y_pred must match.')
