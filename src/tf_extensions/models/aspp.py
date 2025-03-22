"""The module provides the ASPP based segmentation network."""
from dataclasses import dataclass

import tensorflow as tf

from tf_extensions.auxiliary.custom_types import MaskType, TrainingType
from tf_extensions.layers.aspp_layer import ASPPLayer, ASPPLayerConfig
from tf_extensions.models.base_cnn import BaseCNN, BaseCNNConfig


@dataclass
class ASPPNetConfig(BaseCNNConfig):
    """
    Configuration of the ASPPNet model.

    Attributes
    ----------
    aspp_config : ASPPLayerConfig
        Config of Atrous Spatial Pyramid Pooling (ASPP) layer.
    middle_filters : int
        Filters number in the middle convolutional layer.

    """

    aspp_config: ASPPLayerConfig = None
    middle_filters: int = 48

    def __post_init__(self) -> None:
        """Update configuration fields after initialization."""
        if self.aspp_config is None:
            self.aspp_config = ASPPLayerConfig()  # noqa: WPS601

    def get_config_name(self) -> str:
        """
        Return the configuration name based on its attributes.

        Returns
        -------
        str
            A string representation of the configuration.

        """
        return 'aspp{0}middle{1}'.format(
            self.aspp_config.filters_number,
            self.middle_filters,
        )


class ASPPNet(BaseCNN):
    """Atrous Spatial Pyramid Pooling (ASPP) based segmentation network."""

    config_type = ASPPNetConfig

    def __init__(self, **kwargs) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__(**kwargs)
        self.conv_blocks = [
            self.get_convolutional_block(filter_scale=filter_scale)
            for filter_scale in (1, 2, 3, 4, 4, 3)
        ]
        conv2d_kwargs = self.config.conv_block_config.conv2d_config.as_dict()
        self.conv_middle = tf.keras.layers.Conv2D(
            filters=self.config.middle_filters,
            kernel_size=(1, 1),
            **{
                prop_name: prop_value
                for prop_name, prop_value in conv2d_kwargs.items()
                if prop_name != 'kernel_size'
            },
        )
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
            **self.config.aspp_config.as_dict(),
        )

    def call(
        self,
        inputs: tf.Tensor,
        training: TrainingType = None,
        mask: MaskType = None,
    ) -> tf.Tensor:
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        inputs : tf.Tensor
            The input tensor.
        training : TrainingType
            Whether the model is in training mode.
        mask : MaskType
            Mask tensor for specific layers.

        Returns
        -------
        tf.Tensor
            Output tensor.

        """
        out = self.conv_blocks[0](inputs)
        out = self.max_pools[0](out)

        out = self.conv_blocks[1](out)
        out = self.max_pools[1](out)

        out = self.conv_blocks[2](out)
        out_enc_mid = out
        out = self.max_pools[2](out)
        out = self.conv_blocks[3](out)
        out = self.max_pools[3](out)
        out = self.conv_blocks[4](out)

        out = self.aspp(out)

        out = tf.image.resize(
            out,
            tf.shape(out_enc_mid)[1:-1],
            tf.image.ResizeMethod.BILINEAR,
        )

        out_enc_mid = self.conv_middle(out_enc_mid)

        out = tf.concat([out, out_enc_mid], axis=-1)

        out = self.conv_blocks[5](out)
        out = self.conv_out(out)

        out = tf.image.resize(
            out,
            tf.shape(inputs)[1:-1],
            tf.image.ResizeMethod.BILINEAR,
        )
        return tf.nn.sigmoid(out)
