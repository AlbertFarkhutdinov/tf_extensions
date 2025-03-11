import tensorflow as tf


class AttentionGate(tf.keras.layers.Layer):
    """
    Attention Gate layer for focusing on relevant features.

    This layer computes an attention map
    to emphasize important regions in the input tensors.

    Parameters
    ----------
    filters : int
        Number of filters in the intermediate convolutional layers.
    activation : str, optional, default: 'relu'
        Activation function used after feature summation.

    """

    def __init__(
        self,
        filters: int,
        activation: str = 'relu',
        *args,
        **kwargs,
    ) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__(*args, **kwargs)
        self.filters = filters
        conv_kwargs = {
            'kernel_size': (1, 1),
            'padding': 'same',
        }
        self.conv_prev = tf.keras.layers.Conv2D(
            filters=self.filters,
            **conv_kwargs,
        )
        self.conv_skipped = tf.keras.layers.Conv2D(
            filters=self.filters,
            **conv_kwargs,
        )
        self.bn_prev = tf.keras.layers.BatchNormalization()
        self.bn_skipped = tf.keras.layers.BatchNormalization()
        self.activation1 = tf.keras.layers.Activation(activation)
        self.out_layer = tf.keras.layers.Conv2D(
            filters=1,
            **conv_kwargs,
        )
        self.activation2 = tf.keras.layers.Activation('sigmoid')

    def call(self, inputs: list[tf.Tensor], *args, **kwargs) -> tf.Tensor:
        """
        Forward pass of the AttentionGate layer.

        Parameters
        ----------
        inputs : list of tf.Tensor
            A list containing:
                - The feature map from the previous layer.
                - The feature map from the skip connection.

        Returns
        -------
        tf.Tensor
            A tensor representing the attention-weighted output.

        """
        previous, skipped = inputs[0], inputs[1]
        previous = self.conv_prev(previous)
        previous = self.bn_prev(previous)
        skipped = self.conv_skipped(skipped)
        skipped = self.bn_skipped(skipped)
        out = self.activation1(previous + skipped)
        out = self.out_layer(out)
        return self.activation2(out)
