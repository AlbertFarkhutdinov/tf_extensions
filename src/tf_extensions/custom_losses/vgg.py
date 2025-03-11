from dataclasses import dataclass

import tensorflow as tf
from keras.applications.vgg19 import preprocess_input

from tf_extensions.custom_losses.vgg_base import VGGBase, VGGBaseConfig


@dataclass
class VGGLossConfig(VGGBaseConfig):
    name: str = 'vgg'
    loss: str = 'mse'
    filter_size: int = 11
    is_preprocessed: bool = True


class VGGLoss(VGGBase):
    """Class for the VGG loss."""

    config_type = VGGLossConfig

    def _preprocess_images(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
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
        return tf.reduce_mean(
            tf.abs(true_feat - pred_feat),
            axis=[1, 2, 3],
        )
