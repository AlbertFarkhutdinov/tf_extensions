"""The module contains Atrous Spatial Pyramid Pooling (ASPP) layer."""
from dataclasses import dataclass

import tensorflow as tf

from tf_extensions.auxiliary.base_config import BaseConfig
from tf_extensions.layers.base_layer import BaseLayer


@dataclass
class ASPPLayerConfig(BaseConfig):
    """
    Config of Atrous Spatial Pyramid Pooling (ASPP) layer.

    Parameters
    ----------
    filters_number : int
        Number of filters in each convolutional layer.
    dilation_scale : int
        Base scale factor for dilation rates.
    dilation_number : int
        Number of dilated convolutional layers
        (excluding the 1x1 convolution).
    kernel_size : tuple of int, optional, default: (3, 3)
        Kernel size for the dilated convolutional layers.

    """

    filters_number: int = 256
    dilation_scale: int = 6
    dilation_number: int = 3
    kernel_size: tuple[int, ...] = (3, 3)


class ASPPLayer(BaseLayer):
    """
    Atrous Spatial Pyramid Pooling (ASPP) layer.

    This layer applies multiple parallel dilated convolutions
    with different dilation rates to capture multiscale context information
    and then concatenates the outputs.

    Attributes
    ----------
    config: ASPPLayerConfig
        Config of Atrous Spatial Pyramid Pooling (ASPP) layer.

    """

    config_type = ASPPLayerConfig

    def __init__(self, **kwargs) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__(**kwargs)
        self.conv_kwargs = {
            'filters': self.config.filters_number,
            'padding': 'same',
            'activation': 'relu',
        }
        self.dilated_layers = [
            tf.keras.layers.Conv2D(kernel_size=(1, 1), **self.conv_kwargs),
        ]
        for dilation_id in range(self.config.dilation_number):
            dilation_rate = self.config.dilation_scale * (dilation_id + 1)
            self.dilated_layers.append(
                tf.keras.layers.Conv2D(
                    kernel_size=self.config.kernel_size,
                    dilation_rate=dilation_rate,
                    **self.conv_kwargs,
                ),
            )
        self.conv_out = tf.keras.layers.Conv2D(
            kernel_size=(1, 1),
            **self.conv_kwargs,
        )

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        Forward pass of the ASPP layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, height, width, channels).

        Returns
        -------
        tf.Tensor
            Output tensor after applying the ASPP transformation.

        """
        outs = [
            dilated_layer(inputs)
            for dilated_layer in self.dilated_layers
        ]
        out = tf.concat(outs, axis=-1)
        return self.conv_out(out)
