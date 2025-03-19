"""The module contains a block of multiple convolutional layers."""
from typing import Any

import tensorflow as tf

from tf_extensions.layers.base_layer import BaseLayer
from tf_extensions.layers.conv_configs import ConvolutionalBlockConfig


class ConvolutionalBlock(BaseLayer):
    """
    A block of multiple convolutional layers.

    The block can optionally include batch normalization, spatial dropout,
    and skip connections.

    Attributes
    ----------
    config : ConvolutionalBlockConfig
        Configuration of the block.

    """

    config_type = ConvolutionalBlockConfig
    attributes = (
        'conv_layers',
        'normalizations',
        'dropouts',
        'activations',
    )

    def __init__(
        self,
        filters: int,
        **kwargs,
    ) -> None:
        """
        Initialize `ConvolutionalBlock`.

        Parameters
        ----------
        filters : int
            Number of filters in each convolutional layer.

        """
        super().__init__(**kwargs)
        self.filters = filters
        self.conv_layers = []
        self.activations = []
        self.normalizations = []
        self.dropouts = []
        for _ in range(self.config.layers_number):
            self.conv_layers.append(
                tf.keras.layers.Conv2D(
                    filters=self.filters,
                    activation=None,
                    **self.config.conv2d_config.as_dict(),
                ),
            )
            self.activations.append(
                tf.keras.layers.Activation(activation=self.config.activation),
            )
            if self.config.with_bn:
                self.normalizations.append(
                    tf.keras.layers.BatchNormalization(),
                )
        if self.config.with_dropout:
            self.dropouts.append(
                tf.keras.layers.SpatialDropout2D(rate=self.config.drop_rate),
            )
        if self.config.with_skipped:
            conv2d_kwargs = dict(self.config.conv2d_config.as_dict())
            conv2d_kwargs['kernel_size'] = (1, 1)
            self.skipped_connection = tf.keras.layers.Conv2D(
                filters=self.filters,
                activation=None,
                **conv2d_kwargs,
            )

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        Forward pass of the convolutional block.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Output tensor.

        """
        out = inputs
        layers_number = self.config.layers_number
        for layer_id in range(layers_number):
            out = self.conv_layers[layer_id](out)
            if self.config.with_bn:
                out = self.normalizations[layer_id](out)
            if self.config.with_dropout and layer_id == layers_number - 1:
                out = self.dropouts[0](out)
            out = self.activations[layer_id](out)
        if self.config.with_skipped:
            out = out + self.skipped_connection(inputs)
        return out

    def get_config(self) -> dict[str, Any]:
        """
        Return the configuration of the layer.

        Returns
        -------
        dict
            Dictionary containing the layer configuration.

        """
        config = super().get_config()
        for field_name, field_value in self.config.as_dict().items():
            if field_name != 'conv2d_config':
                config[field_name] = field_value

        for attribute in self.__class__.attributes:
            config[attribute] = []
            for layer in getattr(self, attribute):
                config[attribute].append(layer.get_config())

        return config
