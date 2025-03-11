import tensorflow as tf


class AttentionGatingBlock(tf.keras.layers.Layer):
    """
    Attention Gating Block for enhancing feature selection.

    This layer implements an attention mechanism
    that refines feature maps using gating signals.

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
        conv_kwargs = {'padding': 'same'}
        self.conv_prev = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            **conv_kwargs,
        )
        self.conv_skipped = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(2, 2),
            strides=(2, 2),
            **conv_kwargs,
        )
        self.activation1 = tf.keras.layers.Activation(activation)
        self.out_layer = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            **conv_kwargs,
        )
        self.activation2 = tf.keras.layers.Activation('sigmoid')
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
        out = self.activation1(out)
        out = self.out_layer(out)
        out = self.activation2(out)
        return self.up_layer(out)
