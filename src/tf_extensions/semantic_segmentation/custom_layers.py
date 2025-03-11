"""The module provides custom TensorFlow layers for deep learning models."""
from typing import Any, Optional, Union

import numpy as np
import tensorflow as tf

from tf_extensions.semantic_segmentation import configs as cfg

DEFAULT_ACTIVATION = 'relu'
DEFAULT_CONV_KWARGS = {'padding': 'same'}


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
            **DEFAULT_CONV_KWARGS,
            'activation': DEFAULT_ACTIVATION,
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


class ConvolutionalBlock(tf.keras.layers.Layer):
    """
    A block of multiple convolutional layers.

    The block can optionally include batch normalization, spatial dropout,
    and skip connections.

    Parameters
    ----------
    filters : int
        Number of filters in each convolutional layer.
    config : ConvolutionalBlockConfig
        Configuration of the block.

    """

    def __init__(
        self,
        filters: int,
        config: cfg.ConvolutionalBlockConfig,
        *args,
        **kwargs,
    ) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.config = config
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
        config_as_dict = self.config.as_dict()
        for field_name, field_value in config_as_dict.items():
            if field_name != 'conv2d_config':
                config[field_name] = field_value
        attributes = (
            'conv_layers',
            'normalizations',
            'dropouts',
            'activations',
        )
        for attribute in attributes:
            config[attribute] = []
            for layer in getattr(self, attribute):
                config[attribute].append(layer.get_config())

        return config


class MaxPoolingWithArgmax2D(tf.keras.layers.Layer):
    """
    2D Max Pooling layer with Argmax output.

    This layer performs max pooling
    and also returns the indices of the max values,
    which can be useful for operations like unpooling.

    Parameters
    ----------
    pool_size : tuple of int, optional, default: (2, 2)
        Size of the max pooling window.
    strides : tuple of int, optional, default: (2, 2)
        Stride of the pooling operation.
    padding : str, optional, default: 'same'
        Padding method, either 'same' or 'valid'.

    """

    def __init__(
        self,
        pool_size: tuple[int, int] = (2, 2),
        strides: tuple[int, int] = (2, 2),
        padding: str = 'same',
        *args,
        **kwargs,
    ) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__(*args, **kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(
        self,
        inputs: tf.Tensor,
        *args,
        **kwargs,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass of the MaxPoolingWithArgmax2D layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, height, width, channels).

        Returns
        -------
        tuple of tf.Tensor
            - Pooled output tensor.
            - The indices of the max values.

        """
        output, argmax = tf.nn.max_pool_with_argmax(
            input=inputs,
            ksize=[1, self.pool_size[0], self.pool_size[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            padding=self.padding.upper(),
        )
        argmax = tf.cast(argmax, 'int32')
        return output, argmax

    def compute_output_shape(
        self,
        input_shape: tuple[int, ...],
    ) -> list[tuple[int, ...]]:
        """
        Return the output shape of the layer.

        Parameters
        ----------
        input_shape : tuple of int
            Shape of the input tensor.

        Returns
        -------
        list of tuple of int
            - Shape of the pooled output tensor.
            - Shape of the argmax tensor.

        """
        ratio = (1, 2, 2, 1)
        output_shape = []
        for idx, dim in enumerate(input_shape):
            if dim is not None:
                output_shape.append(dim // ratio[idx])
            else:
                output_shape.append(None)
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(
        self,
        inputs: Union[tf.Tensor, list[tf.Tensor]],
        mask: Union[tf.Tensor, list[tf.Tensor]] = None,
    ) -> list[None]:
        """
        Return the mask for the layer.

        Parameters
        ----------
        inputs : tf.Tensor or list of tf.Tensor
            Input tensor(s) for which the mask is computed.
        mask : tf.Tensor or list of tf.Tensor, optional
            Mask tensor(s), if applicable.

        Returns
        -------
        list of None
            A list containing None values, indicating no masking is applied.

        """
        return 2 * [None]


class MaxUnpooling2D(tf.keras.layers.Layer):
    """
    2D Max Unpooling layer for reconstructing feature maps.

    This layer performs the inverse operation of max pooling
    by placing pooled values back into their original positions
    using the provided indices.

    Parameters
    ----------
    pool_size : tuple of int, optional, default: (2, 2)
        Size of the max pooling window.

    """

    def __init__(
        self,
        pool_size: tuple[int, int] = (2, 2),
        *args,
        **kwargs,
    ) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__(*args, **kwargs)
        self.pool_size = pool_size

    def call(
        self,
        inputs: list[tf.Tensor],
        *args,
        **kwargs,
    ) -> tf.Tensor:
        """
        Forward pass of the MaxUnpooling2D layer.

        Parameters
        ----------
        inputs : list of tf.Tensor
            A list containing pooled values and pooling indices.
            Shape of both tensors: (batch_size, height, width, channels).

        Returns
        -------
        tf.Tensor
            The unpooled output tensor with restored spatial dimensions.

        """
        pooling_values, pooling_indices = inputs[0], inputs[1]
        data_type = 'int32'
        pooling_indices = tf.cast(pooling_indices, data_type)
        input_shape = tf.shape(pooling_values, out_type=data_type)

        output_shape = input_shape * np.array([1, *self.pool_size, 1])
        ones_like_indices = tf.ones_like(pooling_indices, dtype=data_type)
        batches = ones_like_indices * tf.reshape(
            tf.range(output_shape[0], dtype=data_type),
            shape=[-1, 1, 1, 1],
        )
        features = ones_like_indices * tf.reshape(
            tf.range(output_shape[-1], dtype=data_type),
            shape=[1, 1, 1, -1],
        )
        rows = pooling_indices // (output_shape[2] * output_shape[3])
        columns = (pooling_indices // output_shape[3]) % output_shape[2]

        scatter = tf.scatter_nd(
            tf.transpose(
                tf.reshape(
                    tf.stack([batches, rows, columns, features]),
                    shape=[4, -1],
                ),
            ),
            tf.keras.backend.flatten(pooling_values),
            shape=output_shape,
        )
        return tf.reshape(
            scatter,
            shape=[
                -1,
                input_shape[1] * self.pool_size[0],
                input_shape[2] * self.pool_size[1],
                input_shape[3],
            ],
        )

    def compute_output_shape(
        self,
        input_shape: list[tuple[int, ...]],
    ) -> tuple[int, ...]:
        """
        Return the output shape of the MaxUnpooling2D layer.

        Parameters
        ----------
        input_shape : list of tuple of int
            A list containing:
            - Shape of the pooled output tensor.
            - Shape of the argmax tensor.

        Returns
        -------
        tuple of int
            The computed output shape after unpooling.

        """
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.pool_size[0],
            mask_shape[2] * self.pool_size[1],
            mask_shape[3],
        )


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
        activation: str = DEFAULT_ACTIVATION,
        *args,
        **kwargs,
    ) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__(*args, **kwargs)
        self.filters = filters
        conv_kwargs = {
            'kernel_size': (1, 1),
            **DEFAULT_CONV_KWARGS,
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


class OutputSkippedConnections(tf.keras.layers.Layer):
    """
    A block of multiple convolutions with skipped connections.

    Output of each convolution block is added with skipped connection
    or concatenates with it.

    Parameters
    ----------
    filters : int
        Number of filters in each convolutional layer.
    config : ConvolutionalBlockConfig, optional
        Configuration of the convolutional block.
        If None, a default configuration is used.
    is_skipped_with_concat : bool, optional, default: True
        If True, skipped connections are concatenated.
        Otherwise, they are summed.
    blocks_number : int, optional, default: 0
        Number of convolutional blocks to apply.

    """

    def __init__(
        self,
        filters: int,
        config: cfg.ConvolutionalBlockConfig = None,
        is_skipped_with_concat: bool = True,
        blocks_number: int = 0,
        **kwargs,
    ) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__(**kwargs)
        self.filters = filters
        if config:
            self.config = config
        else:
            self.config = cfg.ConvolutionalBlockConfig()
        self.is_skipped_with_concat = is_skipped_with_concat
        self.blocks_number = blocks_number
        self.conv_layers = [
            tf.keras.layers.Conv2D(
                filters=self.filters,
                activation=None,
                **self.config.conv2d_config.as_dict(),
            )
            for _ in range(self.blocks_number)
        ]

    def call(
        self,
        inputs: tf.Tensor,
        training: Union[bool, tf.Tensor] = None,
        mask: Union[Optional[tf.Tensor], list[Optional[tf.Tensor]]] = None,
    ) -> tf.Tensor:
        """
        Forward pass of the OutputSkippedConnections block.

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
        outs = [inputs]
        for layer in self.conv_layers:
            out = layer(outs[-1])
            if self.is_skipped_with_concat:
                out = tf.concat([outs[-1], out], axis=-1)
            else:
                out = tf.reduce_sum(input_tensor=[outs[-1], out], axis=0)
            outs.append(out)
        return outs[-1]


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
        conv2d_config: cfg.Conv2DConfig = None,
        **kwargs,
    ) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__(**kwargs)
        self.vector_length = vector_length
        if conv2d_config:
            self.conv2d_config = conv2d_config
        else:
            self.conv2d_config = cfg.Conv2DConfig()
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
        training: Union[bool, tf.Tensor] = None,
        mask: Union[Optional[tf.Tensor], list[Optional[tf.Tensor]]] = None,
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


class GatingSignal(tf.keras.layers.Layer):
    """
    Gating signal layer for attention mechanism.

    This layer processes the gating signal,
    which helps in controlling attention mechanisms
    by applying a 1x1 convolution,
    optional batch normalization, and activation.

    Parameters
    ----------
    filters : int
        Number of filters for the convolutional layer.
    with_bn : bool, optional, default: True
        Whether to apply batch normalization after the convolution.
    activation : str, optional, default: 'relu'
        Activation function applied at the end of the layer.

    """

    def __init__(
        self,
        filters: int,
        with_bn: bool = True,
        activation: str = DEFAULT_ACTIVATION,
        *args,
        **kwargs,
    ) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__(*args, **kwargs)
        self.with_bn = with_bn
        self.filters = filters
        conv_kwargs = {
            'filters': self.filters,
            'kernel_size': 1,
            **DEFAULT_CONV_KWARGS,
        }
        self.convolution = tf.keras.layers.Conv2D(**conv_kwargs)
        self.bn = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation(activation)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        Forward pass of the GatingSignal layer.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor representing the gating signal.

        Returns
        -------
        tf.Tensor
            The processed gating signal tensor after applying convolution,
            optional batch normalization, and activation.

        """
        out = self.convolution(inputs)
        if self.with_bn:
            out = self.bn(out)
        return self.activation(out)


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
        activation: str = DEFAULT_ACTIVATION,
        *args,
        **kwargs,
    ) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__(*args, **kwargs)
        self.filters = filters
        conv_kwargs = {**DEFAULT_CONV_KWARGS}
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
