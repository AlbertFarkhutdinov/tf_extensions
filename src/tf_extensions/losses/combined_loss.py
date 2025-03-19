"""Module with a flexible combination of multiple weighted loss functions."""
from dataclasses import dataclass
from typing import Any, TypeVar

import tensorflow as tf

from tf_extensions.losses.base_loss import BaseLoss, BaseLossConfig
from tf_extensions.losses.supported_losses import supported_losses


@dataclass
class CombinedLossConfig(BaseLossConfig):
    """
    Config of CombinedLoss.

    Parameters
    ----------
    name : str
        Name of combined loss.
    losses : list
        Initialized loss objects.
    weights : list
        Corresponding weights for each loss.

    """

    name: str = ''
    losses: list = None
    weights: list = None

    def __post_init__(self) -> None:
        """
        Update config properties after initialization.

        Raises
        ------
        ValueError
            If losses and weights are not lists with the same length.

        """
        if self.losses is None or self.weights is None:
            msg = 'Losses and weights must be provided as lists.'
            raise ValueError(msg)
        if len(self.losses) != len(self.weights):
            msg = 'Losses and weights lists must have the same length.'
            raise ValueError(msg)
        if not self.name:
            self.name = '_'.join(  # noqa: WPS601
                [loss.name for loss in self.losses],
            )


CombinedLossInstance = TypeVar('CombinedLossInstance', bound='CombinedLoss')


class CombinedLoss(BaseLoss):
    """
    A flexible loss class that combines multiple loss functions with weights.

    Attributes
    ----------
    config : CombinedLossConfig
        Config of CombinedLossConfig.

    """

    config_type = CombinedLossConfig

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """
        Return the loss values.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth values, shape [batch_size, d0, ..., dN].
        y_pred : tf.Tensor
            Predicted values, shape [batch_size, d0, ..., dN].

        Returns
        -------
        tf.Tensor
            The loss function values.

        """
        y_true, y_pred = self.cast_to_dtype(y_true=y_true, y_pred=y_pred)
        weighted_losses = [
            weight * loss_fn(y_true, y_pred)
            for loss_fn, weight in zip(self.config.losses, self.config.weights)
        ]
        return tf.cast(sum(weighted_losses), dtype=self.config.dtype)

    def get_config(self) -> dict[str, Any]:  # noqa: WPS615
        """
        Return the configuration dictionary for serialization.

        Returns
        -------
        dict
            Configuration dictionary containing loss parameters.

        """
        config = super().get_config()
        loss_configs = []
        for loss in self.config.losses:
            loss_configs.append(loss.get_config())
        config['losses'] = loss_configs
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CombinedLossInstance:
        """
        Create a loss instance from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        CombinedLossInstance
            New instance of `CombinedLoss`.

        """
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
