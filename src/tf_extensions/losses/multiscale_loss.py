"""Module with a flexible combination of multiple weighted loss functions."""
from dataclasses import dataclass
from typing import Any, TypeVar

import tensorflow as tf

from tf_extensions.auxiliary import exceptions
from tf_extensions.losses.base_loss import BaseLoss, BaseLossConfig
from tf_extensions.losses.supported_losses import supported_losses


@dataclass
class MultiScaleLossConfig(BaseLossConfig):
    """
    Configuration class for MultiScaleLoss.

    Attributes
    ----------
    name : str
        Name of the loss function.
    base_loss : tf.keras.losses.Loss
        The base loss function to be applied at multiple scales.
    weights : list
        Weights assigned to each scale level.

    """

    name: str = ''
    base_loss: tf.keras.losses.Loss = None
    weights: list = None

    def __post_init__(self) -> None:
        """
        Update config properties after initialization.

        Raises
        ------
        ValueError
            If `base_loss` is not provided.

        """
        if self.base_loss is None:
            raise exceptions.UnspecifiedArgumentError(arg_name='base_loss')
        if not self.name:
            base_loss_name = self.base_loss.name
            self.name = f'multiscale_{base_loss_name}'  # noqa: WPS601


MultiScaleLossInstance = TypeVar(
    'MultiScaleLossInstance',
    bound='MultiScaleLoss',
)


class MultiScaleLoss(BaseLoss):
    """
    A flexible loss class that combines multiple loss functions with weights.

    Attributes
    ----------
    config : MultiScaleLossConfig
        Configuration of MultiScaleLoss.

    """

    config_type = MultiScaleLossConfig

    def call(
        self,
        y_true: tuple[tf.Tensor],
        y_pred: tuple[tf.Tensor],
    ) -> tf.Tensor:
        """
        Return the mean of loss values at multiple scales.

        Parameters
        ----------
        y_true : tuple of tf.Tensor
            Ground truth values at multiple scales.
        y_pred : tuple of tf.Tensor
            Predicted values at multiple scales.

        Returns
        -------
        tf.Tensor
            The computed loss value.

        Raises
        ------
        ValueError
            If inputs are not tuples of tensors with the same length.
            If length of weights in the config and `y_true` do not match.

        """
        if not isinstance(y_true, tuple) or not isinstance(y_pred, tuple):
            raise exceptions.WrongTypeError(
                msg='Inputs must be tuples of tensors.',
            )
        if len(y_true) != len(y_pred):
            raise exceptions.NonEqualValuesError(
                msg='Lengths of y_true and y_pred must match.',
            )
        if self.config.weights and (len(self.config.weights) != len(y_true)):
            raise exceptions.NonEqualValuesError(
                msg='Lengths of weights and y_true must match.',
            )
        batch_size = y_true[0].shape[0]
        losses = []
        level_numbers = len(y_true)
        for level in range(level_numbers):
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
        """
        Return the configuration dictionary for serialization.

        Returns
        -------
        dict
            Configuration dictionary containing loss parameters.

        """
        config = super().get_config()
        config['base_loss'] = self.config.base_loss.get_config()
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MultiScaleLossInstance:
        """
        Create a loss instance from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        MultiScaleLossInstance
            New instance of `MultiScaleLoss`.

        """
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
        """
        Check whether batch sizes of tensors match.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth tensor.
        y_pred : tf.Tensor
            Predicted tensor.
        batch_size : int
            Expected batch size.

        Raises
        ------
        ValueError
            If batch sizes do not match.

        """
        if y_true.shape[0] != batch_size:
            raise exceptions.NonEqualValuesError(
                msg='Batch sizes in y_true must match.',
            )
        if y_pred.shape[0] != batch_size:
            raise exceptions.NonEqualValuesError(
                msg='Batch sizes in y_pred must match.',
            )
