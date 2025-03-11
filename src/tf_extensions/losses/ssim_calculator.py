from dataclasses import dataclass

import tensorflow as tf

from tf_extensions.auxiliary.base_config import BaseConfig


@dataclass
class SSIMCalculatorConfig(BaseConfig):
    """Config of SSIMCalculator."""
    c1: float
    c2: float
    averaging: str


class SSIMCalculator:
    """Class for the luminance and structure calculation."""

    def __init__(
        self,
        tensor1: tf.Tensor,
        tensor2: tf.Tensor,
        averaging_kwargs: dict,
        **kwargs,
    ) -> None:
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
        luminance = self._get_luminance()
        structure = self._get_structure()
        return luminance, structure

    def _get_luminance(self) -> tf.Tensor:
        c1 = self.config.c1
        ms1 = self.ms1
        ms2 = self.ms2
        mean12 = self.mean12
        return (2 * mean12 + c1) / (ms1 + ms2 + c1)

    def _get_structure(self) -> tf.Tensor:
        c2 = self.config.c2
        averaging_kwargs = self.averaging_kwargs
        tensor1 = self.tensor1
        tensor2 = self.tensor2
        var1 = self.averaging(tensor1 * tensor1, **averaging_kwargs) - self.ms1
        var2 = self.averaging(tensor2 * tensor2, **averaging_kwargs) - self.ms2
        cov = self.averaging(
            tensor1 * tensor2,
            **averaging_kwargs,
        ) - self.mean12
        return (2 * cov + c2) / (var1 + var2 + c2)
