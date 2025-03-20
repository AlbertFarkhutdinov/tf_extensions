"""Module providing a class for Soft Dice Loss function."""
from dataclasses import dataclass

import tensorflow as tf

from tf_extensions.losses.base_loss import BaseLoss, BaseLossConfig

STABILIZATION = 1e-9


@dataclass
class SoftDiceLossConfig(BaseLossConfig):
    """
    Configuration class for Soft Dice Loss.

    Attributes
    ----------
    name : str
        Name of the loss function, default is 'sdl'.
    """

    name: str = 'sdl'


class SoftDiceLoss(BaseLoss):
    """
    Class implementing the Soft Dice Loss function.

    This loss function is commonly used for segmentation tasks,
    measuring the overlap between predicted and true labels.
    It is a differentiable approximation of the Dice coefficient,
    which is widely used for evaluating segmentation models.

    Attributes
    ----------
    config : SoftDiceLossConfig
        Configuration of SoftDiceLoss.

    """

    config_type = SoftDiceLossConfig

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """
        Return the SoftDiceLoss loss between the true and predicted images.

        Parameters
        ----------
        y_true : array-like
            Ground truth images.
        y_pred : array-like
            Predicted images.

        Returns
        -------
        float
            The computed SoftDiceLoss loss.

        """
        y_true, y_pred = self.cast_to_dtype(y_true=y_true, y_pred=y_pred)
        prod_sum = self._get_sum(tf.multiply(y_pred, y_true))
        prod_sum *= 2
        sum_true = self._get_sum(tf.pow(y_true, y=2))
        sum_pred = self._get_sum(tf.pow(y_pred, y=2))
        return 1 - tf.divide(prod_sum, sum_true + sum_pred + STABILIZATION)

    def cast_to_dtype(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Cast input tensors to the appropriate dtype and apply thresholding.

        This method ensures that input tensors are cast to the correct dtype
        and binarized using a threshold of 0.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth segmentation mask tensor.
        y_pred : tf.Tensor
            Predicted segmentation mask tensor.

        Returns
        -------
        tuple of tf.Tensor
            Thresholded and cast tensors.

        """
        threshold = 0
        # noinspection PyTypeChecker
        return super().cast_to_dtype(
            y_true=y_true >= threshold,
            y_pred=y_pred >= threshold,
        )

    @classmethod
    def _get_sum(cls, tensor: tf.Tensor) -> tf.Tensor:
        """
        Return the sum of a tensor along spatial and channel axes.

        Parameters
        ----------
        tensor : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Reduced sum over axes [1, 2, 3].

        """
        return tf.reduce_sum(tensor, axis=[1, 2, 3])
