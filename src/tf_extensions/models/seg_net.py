"""The module provides the SegNet network."""
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from tf_extensions.auxiliary.custom_types import MaskType, TrainingType
from tf_extensions.layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from tf_extensions.models.base_net import BaseNet, BaseNetConfig


@dataclass
class SegNetConfig(BaseNetConfig):
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


class SegNet(BaseNet):
    """SegNet network."""

    def __init__(self, **kwargs) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        if 'config' not in kwargs:
            kwargs['config'] = SegNetConfig()
        kwargs['config'].conv_block_config.with_bn = True
        super().__init__(**kwargs)
        self.powers = np.arange(self.config.path_length)
        self.max_pools = []
        self.max_unpools = []
        self.encoder_layers = []
        self.decoder_layers = []

        pooling = self.config.pooling
        padding = self.config.conv_block_config.conv2d_config.padding
        self.output_convolution = tf.keras.layers.Conv2D(
            filters=2,
            kernel_size=(1, 1),
            padding=padding,
        )
        self.output_batch_normalization = tf.keras.layers.BatchNormalization()
        self.output_activation = tf.keras.layers.Activation('softmax')
        for power in self.powers:
            self.max_pools.append(
                MaxPoolingWithArgmax2D(
                    pool_size=(pooling, pooling),
                    strides=(pooling, pooling),
                    padding=padding,
                ),
            )
            self.max_unpools.append(
                MaxUnpooling2D(
                    pool_size=(pooling, pooling),
                ),
            )
            self.encoder_layers.append(
                self.get_convolutional_block(
                    filter_scale=pooling ** power,
                ),
            )
            if power < self.powers[-1]:
                invert_power = self.powers[-1] - power - 1
                self.decoder_layers.append(
                    self.get_convolutional_block(
                        filter_scale=pooling ** invert_power,
                    ),
                )

    def call(
        self,
        inputs: tf.Tensor,
        training: TrainingType = None,
        mask: MaskType = None,
    ) -> tf.Tensor:
        """
        Forward pass of the SegNet model.

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
        output = inputs
        pooling_indices = []
        for enc_power in self.powers:
            output = self.encoder_layers[enc_power](output)
            output, argmax = self.max_pools[enc_power](output)

            pooling_indices.append(argmax)

        for dec_power in self.powers:
            output = self.max_unpools[dec_power](
                [output, pooling_indices[self.powers[-1] - dec_power]],
            )
            if dec_power < self.powers[-1]:
                output = self.decoder_layers[dec_power](output)

        output = self.output_convolution(output)
        output = self.output_batch_normalization(output)
        return self.output_activation(output)
