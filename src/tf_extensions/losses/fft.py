from dataclasses import dataclass

import tensorflow as tf

from tf_extensions.losses.base_loss import BaseLoss, BaseLossConfig


@dataclass
class FFTLossConfig(BaseLossConfig):
    name: str = 'fft'
    loss: str = 'mse'
    filter_size: int = 11
    is_averaged_loss: bool = False

    def __post_init__(self) -> None:
        if self.dtype not in {'float32', 'float64'}:
            raise ValueError(
                'Unsupported dtype in FFTLoss: {0}'.format(self.dtype),
            )


class FFTLoss(BaseLoss):
    """Class for the FFT loss."""

    config_type = FFTLossConfig

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """
        Return the FFT loss between the true and predicted images.

        Parameters
        ----------
        y_true : array-like
            Ground truth images.
        y_pred : array-like
            Predicted images.

        Returns
        -------
        float
            The computed VGG loss.

        """
        fft_true, fft_pred = self._get_fft_pair(y_true=y_true, y_pred=y_pred)
        if self.config.loss == 'ssim':
            ssim = self._get_ssim(fft_true=fft_true, fft_pred=fft_pred)
            return tf.divide(
                tf.convert_to_tensor(value=1, dtype=self.config.dtype) - ssim,
                y=2,
            )

        spectra_difference = fft_true - fft_pred
        if self.config.is_averaged_loss:
            axis = [1, 2, 3]
        else:
            axis = 1
        if self.config.loss == 'mse':
            return tf.reduce_mean(
                tf.square(spectra_difference),
                axis=axis,
            )
        if self.config.loss == 'mae':
            return tf.reduce_mean(
                tf.abs(spectra_difference),
                axis=axis,
            )
        raise ValueError(
            'Unsupported loss function {0}'.format(self.config.loss),
        )

    def _get_fft_pair(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        y_true, y_pred = self.cast_to_dtype(y_true=y_true, y_pred=y_pred)
        y_true, y_pred = self.normalize_images(y_true=y_true, y_pred=y_pred)

        is_xl_averaged = not (
            (self.config.loss == 'ssim') or self.config.is_averaged_loss
        )
        fft_true = self._get_spectra(
            batch=y_true,
            is_xl_averaged=is_xl_averaged,
        )
        fft_pred = self._get_spectra(
            batch=y_pred,
            is_xl_averaged=is_xl_averaged,
        )
        return fft_true, fft_pred

    def _get_ssim(
        self,
        fft_true: tf.Tensor,
        fft_pred: tf.Tensor,
    ) -> tf.Tensor:
        max_true = tf.reduce_max(fft_true)
        try:
            ssim = tf.image.ssim(
                img1=fft_true / max_true,
                img2=fft_pred / max_true,
                max_val=1,
                filter_size=self.config.filter_size,
            )
        except tf.errors.InvalidArgumentError as exc:
            msg = 'Too small image for filter size {0}'.format(
                self.config.filter_size,
            )
            raise ValueError(msg) from exc
        return tf.cast(ssim, self.config.dtype)

    @classmethod
    def _get_spectra(
        cls,
        batch: tf.Tensor,
        is_xl_averaged: bool,
    ) -> tf.Tensor:
        """
        Return spectra of the batch.

        Parameters
        ----------
        batch : array-like
            Input 4D-tensor (batch, time, xline, channels).
        is_xl_averaged : bool
            If it is True, 2D-tensor (batch, frequency) is returned.
            Otherwise, 3D-tensor (batch, frequency, xline) is returned.

        Returns
        -------
        tf.Tensor
            Spectra of the batch.

        """
        transposed = tf.transpose(batch, perm=[0, 3, 2, 1])
        spectra = tf.abs(
            tf.signal.rfft(transposed),
        )
        if is_xl_averaged:
            return tf.reduce_mean(spectra, axis=[1, 2])
        return tf.transpose(spectra, perm=[0, 3, 2, 1])
