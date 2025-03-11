"""The module provides configs for custom layers and models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

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

    kernel_size: tuple[int, int] = (3, 3)
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


@dataclass
class CustomSegmentationNetConfig(BaseConfig):
    """
    Configuration of a custom segmentation network.

    Attributes
    ----------
    conv_block_config : ConvolutionalBlockConfig
        Configuration of convolutional blocks used in the network.
    initial_filters_number : int
        Number of filters in the first convolutional layer.
    max_filters_number : int, optional
        Maximum number of filters allowed in the network.
        If None, no limit is applied.

    """

    conv_block_config: ConvolutionalBlockConfig = field(
        default_factory=ConvolutionalBlockConfig,
    )
    initial_filters_number: int = 16
    max_filters_number: int = None

    def get_config_name(self) -> str:
        """
        Return the configuration name based on its attributes.

        Returns
        -------
        str
            A string representation of the configuration.

        """
        config_name = 'input_neurons{0}'.format(self.initial_filters_number)
        if self.max_filters_number:
            config_name = '{config_name}_max_neurons{max_neurons}'.format(
                config_name=config_name,
                max_neurons=self.max_filters_number,
            )
        return '{config_name}_{conv_block_config_name}'.format(
            config_name=config_name,
            conv_block_config_name=self.conv_block_config.get_config_name(),
        )


@dataclass
class SegNetConfig(CustomSegmentationNetConfig):
    """
    Configuration of the SegNet model.

    Attributes
    ----------
    path_length : int
        The depth of the network (the number of encoder-decoder steps).
    pooling : int
        The down-sampling factor used in pooling layers.

    """

    path_length: int = 4
    pooling: int = 2

    def get_config_name(self) -> str:
        """
        Return the configuration name based on its attributes.

        Returns
        -------
        str
            A string representation of the configuration.

        """
        config_name = 'encoder{0}'.format(self.path_length)
        if self.pooling != 2:
            config_name = '{config_name}_pooling{pooling}'.format(
                config_name=config_name,
                pooling=self.pooling,
            )
        return '{config_name}_{inherited}'.format(
            config_name=config_name,
            inherited=super().get_config_name(),
        )


@dataclass
class UNetConfig(SegNetConfig):
    """
    Configuration of the UNet model.

    Attributes
    ----------
    with_attention : bool, optional, default: False
        Whether to include attention mechanisms in the network.
    without_reducing_filters : bool, optional, default: False
        If True, maintains a constant number of filters while up-sampling.
    is_partial_reducing : bool, optional, default: True
        Whether to apply partial reduction of filters while up-sampling.
    first_blocks_without_dropout : int, optional, default: 0
        Number of initial convolutional blocks without dropout.
    out_residual_blocks_number : int, optional, default: 0
        Number of residual blocks in the output layer.
    is_skipped_with_concat : bool, optional, default: True
        Whether to use concatenation in skip connections.
    with_variable_kernel : bool, optional, default: False
        Whether to allow variable kernel sizes in convolutional layers.
    first_kernel_size : tuple of int, optional
        Kernel size for the first convolutional layer.
    vector_length : int, optional
        Length of the output vector, if 1D output is required.
    is_binary_classification : bool, optional, default: False
        Whether the model is designed for binary classification tasks.

    """

    with_attention: bool = False
    without_reducing_filters: bool = False
    is_partial_reducing: bool = True
    first_blocks_without_dropout: int = 0
    out_residual_blocks_number: int = 0
    is_skipped_with_concat: bool = True
    with_variable_kernel: bool = False
    first_kernel_size: tuple[int, int] = None
    vector_length: int = None
    is_binary_classification: bool = False

    def get_config_name(self) -> str:
        """
        Return the configuration name based on its attributes.

        Returns
        -------
        str
            A string representation of the configuration.

        """
        config_name = ''
        if self.with_attention:
            config_name = '{0}_attention'.format(config_name)
        if self.without_reducing_filters:
            config_name = '{0}_without_reducing_filters'.format(config_name)
            if self.is_partial_reducing:
                config_name = '{0}_partial_reducing'.format(config_name)
        if self.first_blocks_without_dropout:
            config_name = '{config_name}_{blocks_num}without_drop'.format(
                config_name=config_name,
                blocks_num=self.first_blocks_without_dropout,
            )
        if self.out_residual_blocks_number:
            config_name = '{config_name}_out_res{out_res}'.format(
                config_name=config_name,
                out_res=self.out_residual_blocks_number,
            )
            if self.is_skipped_with_concat:
                config_name = '{0}concat'.format(config_name)
            else:
                config_name = '{0}sum'.format(config_name)
        if self.with_variable_kernel:
            config_name = '{0}_with_variable_kernel'.format(config_name)
        if self.first_kernel_size:
            config_name = '{config_name}_first_kernel{ksize1}x{ksize2}'.format(
                config_name=config_name,
                ksize1=self.first_kernel_size[0],
                ksize2=self.first_kernel_size[1],
            )
        if self.vector_length:
            config_name = '{config_name}_vector_length{vector_length}'.format(
                config_name=config_name,
                vector_length=self.vector_length,
            )
        return '{config_name}_{inherited}'.format(
            config_name=config_name,
            inherited=super().get_config_name(),
        ).removeprefix('_')


CustomNetConfigType = Union[
    CustomSegmentationNetConfig,
    SegNetConfig,
    UNetConfig,
]
