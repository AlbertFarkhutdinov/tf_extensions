import tensorflow as tf

from tf_extensions.auxiliary.custom_types import MaskType, TrainingType
from tf_extensions.layers.conv_configs import Conv2DConfig


class UNetOutputLayer(tf.keras.layers.Layer):
    """
    A layer that applies a 1D or 2D convolution to produce the final output.

    Parameters
    ----------
    vector_length : int, optional
        The length of the output vector. If provided, a 1D convolution is used.
        Otherwise, a 2D convolution is used.
    conv2d_config : cfg.Conv2DConfig, optional
        Configuration of the 2D convolution layer.
        If not provided, a default configuration is used.

    """

    def __init__(
        self,
        vector_length: int = None,
        conv2d_config: Conv2DConfig = None,
        **kwargs,
    ) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__(**kwargs)
        self.vector_length = vector_length
        if conv2d_config:
            self.conv2d_config = conv2d_config
        else:
            self.conv2d_config = Conv2DConfig()
        if self.vector_length:
            self.out_layer = tf.keras.layers.Conv1D(
                filters=1,
                kernel_size=self.vector_length,
            )
        else:
            self.out_layer = tf.keras.layers.Conv2D(
                filters=1,
                activation='sigmoid',
                **self.conv2d_config.as_dict(),
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
        training : bool or tf.Tensor, optional
            Whether the layer should behave in training mode or inference mode.
        mask : tf.Tensor or list of tf.Tensor, optional
            Mask tensor(s) for input data.

        Returns
        -------
        tf.Tensor
            Output tensor.

        """
        out = inputs
        if self.vector_length:
            out = tf.image.resize(
                out,
                size=(tf.shape(out)[1], self.vector_length),
                method=tf.image.ResizeMethod.BILINEAR,
            )
            return self.out_layer(out)
        return self.out_layer(out)
