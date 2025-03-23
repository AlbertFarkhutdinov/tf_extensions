"""Module providing frequency-based losses in deep learning models."""
from dataclasses import dataclass

import tensorflow as tf

from tf_extensions.losses.base_loss import BaseLoss, BaseLossConfig


@dataclass
class FFTLossConfig(BaseLossConfig):
    """
    Configuration class of FFTLoss.

    Parameters
    ----------
    name : str, default: 'fft'
        Name of the loss function.
    loss : {'mse', 'mae', 'ssim'}, default: 'mse'
        Type of loss function to use.
    filter_size : int, default: 11
        Filter size used for SSIM calculation.
    is_averaged_loss : bool, default: False
        If True, loss is averaged across multiple axes.

    """

    name: str = 'fft'
    loss: str = 'mse'
    filter_size: int = 11
    is_averaged_loss: bool = False

    def __post_init__(self) -> None:
        """
        Validate attributes after initialization.

        Raises
        ------
        ValueError
            If dtype is not 'float32' or 'float64'.

        """
        if self.dtype not in {'float32', 'float64'}:
            raise ValueError(
                f'Unsupported dtype in FFTLoss: {self.dtype}',
            )


class FFTLoss(BaseLoss):
    """
    Class for computing FFT-based loss functions in deep learning models.

    This class calculates loss functions based on Fourier Transforms of images.
    It supports Mean Squared Error (MSE), Mean Absolute Error (MAE),
    and Structural Similarity Index Measure (SSIM) loss computations
    in the frequency domain.

    Attributes
    ----------
    config : FFTLossConfig
        Configuration of FFTLoss.

    """

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

        Raises
        ------
        ValueError
            If an unsupported loss function is specified.

        """
        loss = self.config.loss
        fft_true, fft_pred = self._get_fft_pair(y_true=y_true, y_pred=y_pred)
        if loss == 'ssim':
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
        if loss == 'mse':
            return tf.reduce_mean(
                tf.square(spectra_difference),
                axis=axis,
            )
        if loss == 'mae':
            return tf.reduce_mean(
                tf.abs(spectra_difference),
                axis=axis,
            )
        raise ValueError(f'Unsupported loss function {loss}')

    def _get_fft_pair(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Return the FFT representations of true and predicted images.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth images.
        y_pred : tf.Tensor
            Predicted images.

        Returns
        -------
        tuple of tf.Tensor
            FFT-transformed true and predicted images.

        """
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
        """
        Return the Structural Similarity Index (SSIM) between FFT images.

        Parameters
        ----------
        fft_true : tf.Tensor
            FFT-transformed ground truth images.
        fft_pred : tf.Tensor
            FFT-transformed predicted images.

        Returns
        -------
        tf.Tensor
            SSIM value between the two images.

        Raises
        ------
        ValueError
            If the filter size is too large for the input image dimensions.

        """
        max_true = tf.reduce_max(fft_true)
        filter_size = self.config.filter_size
        try:
            ssim = tf.image.ssim(
                img1=fft_true / max_true,
                img2=fft_pred / max_true,
                max_val=1,
                filter_size=filter_size,
            )
        except tf.errors.InvalidArgumentError as exc:
            msg = f'Too small image for filter size {filter_size}'
            raise ValueError(msg) from exc
        return tf.cast(ssim, self.config.dtype)

    @classmethod
    def _get_spectra(
        cls,
        batch: tf.Tensor,
        is_xl_averaged: bool,
    ) -> tf.Tensor:
        """
        Return the Fourier spectra of a batch of images.

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
            Fourier spectra of the batch.

        """
        transposed = tf.transpose(batch, perm=[0, 3, 2, 1])
        spectra = tf.abs(
            tf.signal.rfft(transposed),
        )
        if is_xl_averaged:
            return tf.reduce_mean(spectra, axis=[1, 2])
        return tf.transpose(spectra, perm=[0, 3, 2, 1])
