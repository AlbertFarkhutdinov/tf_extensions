"""The module contains a block of skipped connections."""
from dataclasses import dataclass

import tensorflow as tf

from tf_extensions.auxiliary.base_config import BaseConfig
from tf_extensions.auxiliary.custom_types import MaskType, TrainingType
from tf_extensions.layers.base_layer import BaseLayer
from tf_extensions.layers.conv_configs import ConvolutionalBlockConfig


@dataclass
class SkippedConnectionsConfig(BaseConfig):
    """
    Config of a block of multiple convolutions with skipped connections.

    Parameters
    ----------
    filters : int
        Number of filters in each convolutional layer.
    config : ConvolutionalBlockConfig, optional
        Configuration of the convolutional block.
        If None, a default configuration is used.
    is_skipped_with_concat : bool, optional, default: True
        If True, skipped connections are concatenated.
        Otherwise, they are summed.
    blocks_number : int, optional, default: 0
        Number of convolutional blocks to apply.

    """

    filters: int
    config: ConvolutionalBlockConfig = None
    is_skipped_with_concat: bool = True
    blocks_number: int = 0

    def __post_init__(self) -> None:
        """Update config properties after initialization."""
        if self.config is None:
            self.config = ConvolutionalBlockConfig()  # noqa: WPS601


class SkippedConnections(BaseLayer):
    """
    A block of multiple convolutions with skipped connections.

    Output of each convolution block is added with skipped connection
    or concatenates with it.

    Attributes
    ----------
    config: SkippedConnectionsConfig
        Config of SkippedConnections.

    """

    config_type = SkippedConnectionsConfig

    def __init__(self, **kwargs) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__(**kwargs)
        self.conv_layers = [
            tf.keras.layers.Conv2D(
                filters=self.config.filters,
                activation=None,
                **self.config.config.conv2d_config.as_dict(),
            )
            for _ in range(self.config.blocks_number)
        ]

    def call(
        self,
        inputs: tf.Tensor,
        training: TrainingType = None,
        mask: MaskType = None,
    ) -> tf.Tensor:
        """
        Forward pass of the OutputSkippedConnections block.

        Parameters
        ----------
        inputs : tf.Tensor
            The input tensor.
        training : TrainingType
            Whether the layer should behave in training mode or inference mode.
        mask : MaskType
            Mask tensor(s) for input data.

        Returns
        -------
        tf.Tensor
            Output tensor.

        """
        outs = [inputs]
        for layer in self.conv_layers:
            out = layer(outs[-1])
            if self.config.is_skipped_with_concat:
                out = tf.concat([outs[-1], out], axis=-1)
            else:
                out = tf.reduce_sum(input_tensor=[outs[-1], out], axis=0)
            outs.append(out)
        return outs[-1]
