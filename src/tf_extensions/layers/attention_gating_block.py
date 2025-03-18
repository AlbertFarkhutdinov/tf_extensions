"""The module contains Attention Gating Block."""
from dataclasses import dataclass

import tensorflow as tf

from tf_extensions.auxiliary.base_config import BaseConfig
from tf_extensions.layers.base_layer import BaseLayer


@dataclass
class AttentionGatingBlockConfig(BaseConfig):
    """
    Config of Attention Gating Block for enhancing feature selection.

    Parameters
    ----------
    filters : int
        Number of filters in the intermediate convolutional layers.
    activation : str, optional, default: 'relu'
        Activation function used after feature summation.

    """

    filters: int
    activation: str = 'relu'


class AttentionGatingBlock(BaseLayer):
    """
    Attention Gating Block for enhancing feature selection.

    This layer implements an attention mechanism
    that refines feature maps using gating signals.

    Attributes
    ----------
    config: AttentionGatingBlockConfig
        Config of Attention Gating Block.

    """

    config_type = AttentionGatingBlockConfig

    def __init__(self, **kwargs) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__(**kwargs)
        conv_kwargs = {'padding': 'same'}
        self.conv_prev = tf.keras.layers.Conv2D(
            filters=self.config.filters,
            kernel_size=(1, 1),
            **conv_kwargs,
        )
        self.conv_skipped = tf.keras.layers.Conv2D(
            filters=self.config.filters,
            kernel_size=(2, 2),
            strides=(2, 2),
            **conv_kwargs,
        )
        self.activations = [
            tf.keras.layers.Activation(
                activation=self.config.activation,
            ),
            tf.keras.layers.Activation('sigmoid'),
        ]
        self.out_layer = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            **conv_kwargs,
        )
        self.up_layer = tf.keras.layers.UpSampling2D(size=(2, 2))
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs: list[tf.Tensor], *args, **kwargs) -> tf.Tensor:
        """
        Forward pass of the AttentionGatingBlock.

        Parameters
        ----------
        inputs : list of tf.Tensor
            A list containing:
                - The feature map from the skip connection.
                - The feature map from the previous layer.

        Returns
        -------
        tf.Tensor
            A tensor representing the refined attention-weighted feature map.

        """
        skipped, previous = inputs[0], inputs[1]
        theta_skipped = self.conv_skipped(skipped)
        phi_prev = self.conv_prev(previous)
        out = tf.keras.layers.add([phi_prev, theta_skipped])
        out = self.activations[0](out)
        out = self.out_layer(out)
        out = self.activations[1](out)
        return self.up_layer(out)
