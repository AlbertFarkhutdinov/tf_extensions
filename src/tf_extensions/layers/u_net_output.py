"""The module contains U-Net Output Layer."""
from dataclasses import dataclass

import tensorflow as tf

from tf_extensions.auxiliary.base_config import BaseConfig
from tf_extensions.auxiliary.custom_types import MaskType, TrainingType
from tf_extensions.layers.base_layer import BaseLayer
from tf_extensions.layers.conv_configs import Conv2DConfig


@dataclass
class UNetOutputLayerConfig(BaseConfig):
    """
    Config of U-Net Output Layer.

    Parameters
    ----------
    vector_length : int, optional
        The length of the output vector. If provided, a 1D convolution is used.
        Otherwise, a 2D convolution is used.
    conv2d_config : cfg.Conv2DConfig, optional
        Configuration of the 2D convolution layer.
        If not provided, a default configuration is used.

    """

    vector_length: int = None
    conv2d_config: Conv2DConfig = None

    def __post_init__(self) -> None:
        """Update config properties after initialization."""
        if self.conv2d_config is None:
            self.conv2d_config = Conv2DConfig()  # noqa: WPS601


class UNetOutputLayer(BaseLayer):
    """
    A layer that applies a 1D or 2D convolution to produce the final output.

    Attributes
    ----------
    config: UNetOutputLayerConfig
        Config of UNetOutputLayer.

    """

    config_type = UNetOutputLayerConfig

    def __init__(self, **kwargs) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__(**kwargs)
        if self.config.vector_length:
            self.out_layer = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=self.config.vector_length,
            )
        else:
            self.out_layer = tf.keras.layers.Conv2D(
                filters=1,
                activation='sigmoid',
                **self.config.conv2d_config.as_dict(),
            )

    def call(
        self,
        inputs: tf.Tensor,
        training: TrainingType = None,
        mask: MaskType = None,
    ) -> tf.Tensor:
        """
        Forward pass of the U-Net output layer.

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
        out = inputs
        if self.config.vector_length:
            out = tf.image.resize(
                out,
                size=(tf.shape(out)[1], self.config.vector_length),
                method=tf.image.ResizeMethod.BILINEAR,
            )
            return self.out_layer(out)
        return self.out_layer(out)
