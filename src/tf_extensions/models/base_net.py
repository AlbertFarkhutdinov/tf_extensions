"""The module provides a custom segmentation network."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import tensorflow as tf
from IPython import display

from tf_extensions.auxiliary.base_config import BaseConfig
from tf_extensions.auxiliary.custom_types import MaskType, TrainingType
from tf_extensions.layers import ConvolutionalBlock
from tf_extensions.layers import conv_configs as cc


@dataclass
class BaseNetConfig(BaseConfig):
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


class BaseNet(tf.keras.Model):
    """
    A custom segmentation network.

    Parameters
    ----------
    config : cfg.CustomNetConfigType, optional
        Configuration of the network.
        If None, a default configuration is used.
    include_top : bool, optional, default: True
        Whether to include the final classification layer.

    """

    def __init__(
        self,
        config=None,
        include_top: bool = True,
    ) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__()
        self.include_top = include_top
        self.config = config or BaseNetConfig()
        kernel_size = self.config.conv_block_config.conv2d_config.kernel_size
        if not kernel_size[0] % 2 or not kernel_size[1] % 2:
            raise ValueError('Odd `kernel_size` is recommended.')

    def call(
        self,
        inputs: tf.Tensor,
        training: TrainingType = None,
        mask: MaskType = None,
    ) -> tf.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        inputs : tf.Tensor
            The input tensor.
        training : bool or tf.Tensor, optional
            Whether the model is in training mode.
        mask : tf.Tensor or list of tf.Tensor, optional
            Mask tensor for specific layers.

        Returns
        -------
        tf.Tensor
            Output tensor.

        """
        return inputs

    def build_graph(self, input_shape: tuple[int, int, int]) -> tf.keras.Model:
        """
        Build a Keras model graph based on the specified input shape.

        Parameters
        ----------
        input_shape : tuple of int
            Shape of the input tensor (height, width, channels).

        Returns
        -------
        tf.keras.Model
            A Keras model instance.

        """
        input_layer = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(
            inputs=[input_layer],
            outputs=self.call(input_layer),
        )

    def plot(
        self,
        input_shape: tuple[int, int, int],
        *args,
        **kwargs,
    ) -> display.Image:
        """
        Plot the model architecture.

        Parameters
        ----------
        input_shape : tuple of int
            Shape of the input tensor (height, width, channels).

        Returns
        -------
        display.Image
            Representation of the model architecture.

        """
        return tf.keras.utils.plot_model(
            self.build_graph(input_shape),
            *args,
            show_shapes=True,
            to_file='{0}.png'.format(self.__class__.__name__),
            **kwargs,
        )

    def summary(self, *args, **kwargs) -> None:
        """Print the summary of the model architecture."""
        return self.build_graph(   # pragma: no cover
            input_shape=kwargs.pop('input_shape'),
        ).summary(*args, **kwargs)

    def get_convolutional_block(
        self,
        filter_scale: int,
        kernel_size: tuple[int, int] = None,
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
        is_dropout_off : bool, optional, default: False
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
