"""The module provides configs for custom layers."""
from __future__ import annotations

from dataclasses import dataclass, field

from tf_extensions.auxiliary.base_config import BaseConfig


@dataclass
class Conv2DConfig(BaseConfig):
    """
    Configuration of a 2D convolutional layer.

    Attributes
    ----------
    kernel_size : tuple of int, default: (3, 3)
        Size of the convolutional kernel.
    padding : str, default: 'same'
        Padding mode ('same' or 'valid').
    use_bias : bool, default: True
        Whether to include a bias term in the convolution.
    kernel_initializer : str, default: 'glorot_uniform'
        Initialization method for kernel weights.

    """

    kernel_size: tuple[int, ...] = (3, 3)
    padding: str = 'same'
    use_bias: bool = True
    kernel_initializer: str = 'glorot_uniform'

    def get_config_name(self) -> str:
        """
        Return the configuration name based on its attributes.

        Returns
        -------
        str
            A string representation of the configuration.

        """
        config_name = 'kernel{ksize1}x{ksize2}'.format(
            ksize1=self.kernel_size[0],
            ksize2=self.kernel_size[1],
        )
        if self.padding != 'same':
            config_name = '{0}_pad_{1}'.format(config_name, self.padding)
        if not self.use_bias:
            config_name = '{0}_without_bias'.format(config_name)
        if self.kernel_initializer != 'glorot_uniform':
            config_name = '{config_name}_init_{kernel_initializer}'.format(
                config_name=config_name,
                kernel_initializer=self.kernel_initializer,
            )
        return config_name


@dataclass
class ConvolutionalBlockConfig(BaseConfig):
    """
    Configuration of a convolutional block.

    Attributes
    ----------
    conv2d_config : Conv2DConfig
        Configuration of the Conv2D layers.
    layers_number : int, default: 2
        Number of convolutional layers.
    activation : str, default: 'relu'
        Activation function to use.
    with_skipped : bool, default: False
        Whether to include skip connections.
    with_bn : bool, default: False
        Whether to include batch normalization.
    with_dropout : bool, default: False
        Whether to include dropout.
    drop_rate : float, default: 0.5
        Dropout rate if dropout is enabled.

    """

    conv2d_config: Conv2DConfig = field(default_factory=Conv2DConfig)
    layers_number: int = 2
    activation: str = 'relu'
    with_skipped: bool = False
    with_bn: bool = False
    with_dropout: bool = False
    drop_rate: float = 0.5

    def get_config_name(self) -> str:
        """
        Return the configuration name based on its attributes.

        Returns
        -------
        str
            A string representation of the configuration.

        """
        config_name = self.activation + str(self.layers_number)
        if self.with_skipped:
            config_name = '{0}_residual'.format(config_name)
        if self.with_bn:
            config_name = '{0}_bn'.format(config_name)
        if self.with_dropout:
            config_name = '{config_name}_drop{drop_rate}'.format(
                config_name=config_name,
                drop_rate=int(round(self.drop_rate * 100)),
            )
        return '{config_name}_{conv2d_config_name}'.format(
            config_name=config_name,
            conv2d_config_name=self.conv2d_config.get_config_name(),
        )
