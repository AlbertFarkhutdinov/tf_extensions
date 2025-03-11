import tensorflow as tf


class ASPPLayer(tf.keras.layers.Layer):
    """
    Atrous Spatial Pyramid Pooling (ASPP) layer for semantic segmentation.

    This layer applies multiple parallel dilated convolutions
    with different dilation rates to capture multiscale context information
    and then concatenates the outputs.

    Parameters
    ----------
    filters_number : int
        Number of filters in each convolutional layer.
    dilation_scale : int
        Base scale factor for dilation rates.
    dilation_number : int, optional, default: 3
        Number of dilated convolutional layers (excluding the 1x1 convolution).
    kernel_size : tuple of int, optional, default: (3, 3)
        Kernel size for the dilated convolutional layers.

    """

    def __init__(
        self,
        filters_number: int,
        dilation_scale: int,
        dilation_number: int = 3,
        kernel_size: tuple[int, int] = (3, 3),
        *args,
        **kwargs,
    ) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__(*args, **kwargs)
        self.conv_kwargs = {
            'filters': filters_number,
            'padding': 'same',
            'activation': 'relu',
        }
        self.dilated_layers = [
            tf.keras.layers.Conv2D(kernel_size=(1, 1), **self.conv_kwargs),
        ]
        for dilation_id in range(dilation_number):
            self.dilated_layers.append(
                tf.keras.layers.Conv2D(
                    kernel_size=kernel_size,
                    dilation_rate=dilation_scale * (dilation_id + 1),
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
