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

    def __post_init__(self) -> None:
        """
        Validate attributes after initialization.

        Raises
        ------
        ValueError
            If `kernel_size` contains even elements.

        """
        kernel_size = self.kernel_size
        if not kernel_size[0] % 2 or not kernel_size[1] % 2:
            raise ValueError('Odd `kernel_size` is recommended.')

    def get_config_name(self) -> str:
        """
        Return the configuration name based on its attributes.

        Returns
        -------
        str
            A string representation of the configuration.

        """
        kernel_size = self.kernel_size
        name_parts = [
            f'kernel{kernel_size[0]}x{kernel_size[1]}',
        ]
        if self.padding != 'same':
            name_parts.append(f'pad_{self.padding}')
        if not self.use_bias:
            name_parts.append('without_bias')
        if self.kernel_initializer != 'glorot_uniform':
            name_parts.append(f'init_{self.kernel_initializer}')
        return '_'.join(name_parts)


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
        name_parts = [self.activation + str(self.layers_number)]
        if self.with_skipped:
            name_parts.append('residual')
        if self.with_bn:
            name_parts.append('bn')
        if self.with_dropout:
            drop_rate_percent = int(round(self.drop_rate * 100))
            name_parts.append(f'drop{drop_rate_percent}')
        name_parts.append(self.conv2d_config.get_config_name())
        return '_'.join(name_parts)
