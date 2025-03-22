"""The module provides a custom convolutional neural network (CNN)."""
from copy import deepcopy
from dataclasses import dataclass, field

from tf_extensions.layers import ConvolutionalBlock
from tf_extensions.layers import conv_configs as cc
from tf_extensions.models.base_net import BaseNet, BaseNetConfig


@dataclass
class BaseCNNConfig(BaseNetConfig):
    """
    Configuration of a custom convolutional neural network (CNN).

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

    conv_block_config: cc.ConvolutionalBlockConfig = field(
        default_factory=cc.ConvolutionalBlockConfig,
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

        name_parts = [
            super().get_config_name(),
            'input_neurons{0}'.format(self.initial_filters_number),
        ]
        if self.max_filters_number:
            name_parts.append(
                'max_neurons{0}'.format(self.max_filters_number),
            )
        name_parts.append(self.conv_block_config.get_config_name())
        return '_'.join(name_parts)


class BaseCNN(BaseNet):
    """
    A custom convolutional neural network (CNN).

    Attributes
    ----------
    config : BaseCNNConfig
        Configuration of the model.

    """

    config_type = BaseCNNConfig

    def get_convolutional_block(
        self,
        filter_scale: int,
        kernel_size: tuple[int, ...] = None,
        is_dropout_off: bool = False,
    ) -> ConvolutionalBlock:
        """
        Return a convolutional block with the specified configuration.

        Parameters
        ----------
        filter_scale : int
            Scale factor for the number of filters in the convolutional layers.
        kernel_size : tuple of int, optional
            Kernel size for the convolutional layers.
            If None, the default is used.
        is_dropout_off : bool
            Whether to disable dropout in the convolutional block.

        Returns
        -------
        ConvolutionalBlock
            A configured convolutional block instance.

        """
        config = deepcopy(self.config.conv_block_config)
        if kernel_size is not None:
            config.conv2d_config.kernel_size = kernel_size
        if is_dropout_off:
            config.with_dropout = False
        filters = self.config.initial_filters_number * filter_scale
        if self.config.max_filters_number:
            filters = min(self.config.max_filters_number, filters)
        return ConvolutionalBlock(
            filters=filters,
            config=config,
        )
