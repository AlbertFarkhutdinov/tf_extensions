"""The module provides the ASPP based segmentation network."""
from typing import Optional, Union

import tensorflow as tf

from tf_extensions.semantic_segmentation.custom_layers import ASPPLayer
from tf_extensions.semantic_segmentation.custom_net import CustomSegmentationNet

DEFAULT_MIDDLE_FILTERS = 48
DEFAULT_ASPP_FILTERS = 256


class ASPPNet(CustomSegmentationNet):
    """Atrous Spatial Pyramid Pooling (ASPP) based segmentation network."""

    def __init__(self, **kwargs) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__(**kwargs)

        self.conv_pair1 = self.get_convolutional_block(filter_scale=1)
        self.conv_pair2 = self.get_convolutional_block(filter_scale=2)
        self.conv_pair3 = self.get_convolutional_block(filter_scale=3)
        self.conv_pair4 = self.get_convolutional_block(filter_scale=4)
        self.conv_pair5 = self.get_convolutional_block(filter_scale=4)

        conv2d_kwargs = self.config.conv_block_config.conv2d_config.as_dict()
        self.conv_middle = tf.keras.layers.Conv2D(
            filters=DEFAULT_MIDDLE_FILTERS,
            kernel_size=(1, 1),
            **{
                prop_name: prop_value
                for prop_name, prop_value in conv2d_kwargs.items()
                if prop_name != 'kernel_size'
            },
        )

        self.conv_pair6 = self.get_convolutional_block(filter_scale=3)
        self.conv_out = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            padding='same',
            activation=None,
        )
        self.max_pools = [
            tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same',
            )
            for _ in range(4)
        ]
        self.aspp = ASPPLayer(
            filters_number=DEFAULT_ASPP_FILTERS,
            dilation_scale=6,
            dilation_number=3,
            kernel_size=conv2d_kwargs['kernel_size'],
        )

    def call(
        self,
        inputs: tf.Tensor,
        training: Union[bool, tf.Tensor] = None,
        mask: Union[Optional[tf.Tensor], list[Optional[tf.Tensor]]] = None,
    ) -> tf.Tensor:
        """
        Forward pass of the ASPPNet model.

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
        out = self.conv_pair1(inputs)
        out = self.max_pools[0](out)

        out = self.conv_pair2(out)
        out = self.max_pools[1](out)

        out = self.conv_pair3(out)
        out_enc_mid = out
        out = self.max_pools[2](out)
        out = self.conv_pair4(out)
        out = self.max_pools[3](out)
        out = self.conv_pair5(out)

        out = self.aspp(out)

        out = tf.image.resize(
            out,
            tf.shape(out_enc_mid)[1:-1],
            tf.image.ResizeMethod.BILINEAR,
        )

        out_enc_mid = self.conv_middle(out_enc_mid)

        out = tf.concat([out, out_enc_mid], axis=-1)

        out = self.conv_pair6(out)
        out = self.conv_out(out)

        out = tf.image.resize(
            out,
            tf.shape(inputs)[1:-1],
            tf.image.ResizeMethod.BILINEAR,
        )
        return tf.nn.sigmoid(out)
