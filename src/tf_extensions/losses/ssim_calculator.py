"""
Module containing the SSIMCalculator class.

SSIMCalculator is a class for computing the luminance and structure components
of the Structural Similarity Index (SSIM) between two tensors.
It is also used to calculate texture and structure components of DISTS loss.

"""
from dataclasses import dataclass

import tensorflow as tf

from tf_extensions.auxiliary.base_config import BaseConfig


@dataclass
class SSIMCalculatorConfig(BaseConfig):
    """
    Configuration class for SSIMCalculator.

    Attributes
    ----------
    c1 : float
        Constant for luminance component stabilization.
    c2 : float
        Constant for structure component stabilization.
    averaging : str
        Method used for averaging ('conv2d' or 'reduce_mean').

    """

    c1: float
    c2: float
    averaging: str


class SSIMCalculator:
    """
    Class for the luminance and structure calculation.

    Attributes
    ----------
    config : SSIMCalculatorConfig
        Configuration of SSIMCalculator.

    """

    def __init__(
        self,
        tensor1: tf.Tensor,
        tensor2: tf.Tensor,
        averaging_kwargs: dict,
        **kwargs,
    ) -> None:
        """
        Initialize SSIMCalculator.

        Parameters
        ----------
        tensor1 : tf.Tensor
            First input tensor.
        tensor2 : tf.Tensor
            Second input tensor.
        averaging_kwargs : dict
            Additional arguments for the averaging function.

        """
        self.config = SSIMCalculatorConfig.from_kwargs(**kwargs)
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.averaging_kwargs = averaging_kwargs
        if self.config.averaging == 'conv2d':
            self.averaging = tf.nn.depthwise_conv2d
        elif self.config.averaging == 'reduce_mean':
            self.averaging = tf.reduce_mean
        mean1 = self.averaging(tensor1, **self.averaging_kwargs)
        mean2 = self.averaging(tensor2, **self.averaging_kwargs)
        self.mean12 = mean1 * mean2
        self.ms1 = mean1 * mean1
        self.ms2 = mean2 * mean2

    def calculate(self) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Return the luminance and structure components of SSIM.

        Returns
        -------
        tuple of tf.Tensor
            Luminance and structure components as tensors.

        """
        luminance = self._get_luminance()
        structure = self._get_structure()
        return luminance, structure

    def _get_luminance(self) -> tf.Tensor:
        """
        Return the luminance component of SSIM.

        Returns
        -------
        tf.Tensor
            Luminance component tensor.

        """
        c1 = self.config.c1
        ms1 = self.ms1
        ms2 = self.ms2
        mean12 = self.mean12
        return (2 * mean12 + c1) / (ms1 + ms2 + c1)

    def _get_structure(self) -> tf.Tensor:
        """
        Return the structure component of SSIM.

        Returns
        -------
        tf.Tensor
            Structure component tensor.

        """
        c2 = self.config.c2
        var1 = self._get_cov(
            tensor1=self.tensor1,
            tensor2=self.tensor1,
            mean_value=self.ms1,
        )
        var2 = self._get_cov(
            tensor1=self.tensor2,
            tensor2=self.tensor2,
            mean_value=self.ms2,
        )
        cov = self._get_cov(
            tensor1=self.tensor1,
            tensor2=self.tensor2,
            mean_value=self.mean12,
        )
        cov *= 2
        return (cov + c2) / (var1 + var2 + c2)

    def _get_cov(
        self,
        tensor1: tf.Tensor,
        tensor2: tf.Tensor,
        mean_value: tf.Tensor,
    ) -> tf.Tensor:
        """
        Return covariance of two tensors.

        Parameters
        ----------
        tensor1 : tf.Tensor
            First input tensor.
        tensor2 : tf.Tensor
            Second input tensor.
        mean_value : tf.Tensor
            Mean of tensor product.

        Returns
        -------
        tf.Tensor
            Covariance of two tensors.

        """
        return self.averaging(
            tensor1 * tensor2,
            **self.averaging_kwargs,
        ) - mean_value
