"""Module providing a VGG-based perceptual loss function using TensorFlow."""

from dataclasses import dataclass

import tensorflow as tf
from keras.applications.vgg19 import preprocess_input

from tf_extensions.losses.vgg_base import VGGBase, VGGBaseConfig


@dataclass
class VGGLossConfig(VGGBaseConfig):
    """
    Configuration for the VGGLoss class.

    Attributes
    ----------
    name : str
        Name of the loss function.
    loss : str
        Type of loss function to use ('mse', 'mae', or 'ssim').
    filter_size : int
        Filter size for SSIM computation.
    is_preprocessed : bool
        Whether input images are required preprocessing for VGG.

    """

    name: str = 'vgg'
    loss: str = 'mse'
    filter_size: int = 11
    is_preprocessed: bool = True


class VGGLoss(VGGBase):
    """
    Class for computing perceptual loss using VGG feature maps.

    This class extends `VGGBase` to compute loss based on feature maps
    extracted from a VGG model.
    It supports multiple loss functions, including MSE, MAE, and DSSIM.

    Attributes
    ----------
    config : VGGLossConfig
        Configuration of VGGLoss.

    """

    config_type = VGGLossConfig

    def _preprocess_images(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Preprocesses input images before extracting VGG feature maps.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground-truth images.
        y_pred : tf.Tensor
            Predicted images.

        Returns
        -------
        tuple of tf.Tensor
            Preprocessed ground-truth and predicted images.

        """
        y_true, y_pred = super()._preprocess_images(
            y_true=y_true,
            y_pred=y_pred,
        )
        if self.config.is_preprocessed:
            y_true, y_pred = self._preprocess_for_vgg19(
                y_true=y_true,
                y_pred=y_pred,
            )
        return y_true, y_pred

    @classmethod
    def _preprocess_for_vgg19(
        cls,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Prepare images for VGG19.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground-truth images.
        y_pred : tf.Tensor
            Predicted images.

        Returns
        -------
        tuple of tf.Tensor
            Preprocessed ground-truth and predicted images.

        Raises
        ------
        ValueError
            If images do not have exactly three channels.

        """
        true_channels = y_true.shape[-1]
        pred_channels = y_pred.shape[-1]
        if true_channels != 3:
            msg = 'True image has {0} channels. Required: 3.'.format(
                true_channels,
            )
            raise ValueError(msg)
        if pred_channels != 3:
            msg = 'Predicted image has {0} channels. Required: 3.'.format(
                pred_channels,
            )
            raise ValueError(msg)
        y_true = preprocess_input(y_true * tf.uint8.max)
        y_pred = preprocess_input(y_pred * tf.uint8.max)
        return y_true, y_pred

    def _compute_loss(
        self,
        true_features: tf.Tensor,
        pred_features: tf.Tensor,
    ) -> tf.Tensor:
        """
        Return the loss between true and predicted feature maps.

        Parameters
        ----------
        true_features : tf.Tensor
            Feature maps extracted from VGG for ground-truth images.
        pred_features : tf.Tensor
            Feature maps extracted from VGG for predicted images.

        Returns
        -------
        tf.Tensor
            Computed loss value.

        Raises
        ------
        ValueError
            If the specified loss function is not supported.

        """
        loss_methods = {
            'mse': self._get_mse,
            'mae': self._get_mae,
            'ssim': self._get_dssim,
        }
        if self.config.loss not in loss_methods:
            raise ValueError(
                'Unsupported loss function {0}'.format(self.config.loss),
            )
        losses = []
        for true_feat, pred_feat in zip(true_features, pred_features):
            true_feat = tf.cast(true_feat, self.config.dtype)
            pred_feat = tf.cast(pred_feat, self.config.dtype)
            loss = loss_methods[self.config.loss](
                true_feat=true_feat,
                pred_feat=pred_feat,
            )
            losses.append(loss)
        return tf.reduce_sum(losses, axis=0)

    def _get_dssim(
        self,
        true_feat: tf.Tensor,
        pred_feat: tf.Tensor,
    ) -> tf.Tensor:
        """
        Return the DSSIM loss.

        Parameters
        ----------
        true_feat : tf.Tensor
            Feature map of the ground-truth image.
        pred_feat : tf.Tensor
            Feature map of the predicted image.

        Returns
        -------
        tf.Tensor
            Computed DSSIM loss.

        Raises
        ------
        ValueError
            If filter size in config is too big for the specified VGG layer.

        """
        if self.config.is_preprocessed:
            max_value = tf.uint8.max
        else:
            max_value = 2
        try:
            ssim = tf.image.ssim(
                img1=true_feat,
                img2=pred_feat,
                max_val=max_value,
                filter_size=self.config.filter_size,
            )
        except tf.errors.InvalidArgumentError as exc:
            msg = 'Too big filter size for the specified VGG layer.'
            raise ValueError(msg) from exc
        ssim = tf.cast(ssim, self.config.dtype)
        return (1 - ssim) / 2

    @classmethod
    def _get_mse(
        cls,
        true_feat: tf.Tensor,
        pred_feat: tf.Tensor,
    ) -> tf.Tensor:
        """
        Return the MSE loss.

        Parameters
        ----------
        true_feat : tf.Tensor
            Feature map of the ground-truth image.
        pred_feat : tf.Tensor
            Feature map of the predicted image.

        Returns
        -------
        tf.Tensor
            Computed MSE loss.

        """
        return tf.reduce_mean(
            tf.square(true_feat - pred_feat),
            axis=[1, 2, 3],
        )

    @classmethod
    def _get_mae(
        cls,
        true_feat: tf.Tensor,
        pred_feat: tf.Tensor,
    ) -> tf.Tensor:
        """
        Return the MAE loss.

        Parameters
        ----------
        true_feat : tf.Tensor
            Feature map of the ground-truth image.
        pred_feat : tf.Tensor
            Feature map of the predicted image.

        Returns
        -------
        tf.Tensor
            Computed MAE loss.

        """
        return tf.reduce_mean(
            tf.abs(true_feat - pred_feat),
            axis=[1, 2, 3],
        )
