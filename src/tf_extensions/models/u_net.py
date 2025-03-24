"""The module provides the U-Net network."""
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from tf_extensions import layers as cl
from tf_extensions.auxiliary.custom_types import MaskType, TrainingType
from tf_extensions.auxiliary.exceptions import EvenKernelError
from tf_extensions.models.base_cnn import BaseCNN
from tf_extensions.models.seg_net import SegNetConfig


@dataclass
class UNetConfig(SegNetConfig):
    """
    Configuration of the UNet model.

    Attributes
    ----------
    with_attention : bool, optional, default: False
        Whether to include attention mechanisms in the network.
    without_reducing_filters : bool, optional, default: False
        If True, maintains a constant number of filters while up-sampling.
    is_partial_reducing : bool, optional, default: True
        Whether to apply partial reduction of filters while up-sampling.
    first_blocks_without_dropout : int, optional, default: 0
        Number of initial convolutional blocks without dropout.
    out_residual_blocks_number : int, optional, default: 0
        Number of residual blocks in the output layer.
    is_skipped_with_concat : bool, optional, default: True
        Whether to use concatenation in skip connections.
    with_variable_kernel : bool, optional, default: False
        Whether to allow variable kernel sizes in convolutional layers.
    first_kernel_size : tuple of int, optional
        Kernel size for the first convolutional layer.
    vector_length : int, optional
        Length of the output vector, if 1D output is required.
    is_binary_classification : bool, optional, default: False
        Whether the model is designed for binary classification tasks.

    """

    with_attention: bool = False
    without_reducing_filters: bool = False
    is_partial_reducing: bool = True
    first_blocks_without_dropout: int = 0
    out_residual_blocks_number: int = 0
    is_skipped_with_concat: bool = True
    with_variable_kernel: bool = False
    first_kernel_size: tuple[int, ...] = None
    vector_length: int = None
    is_binary_classification: bool = False

    def __post_init__(self) -> None:
        """
        Validate attributes after initialization.

        Raises
        ------
        ValueError
            If `first_kernel_size` contains even elements.

        """
        super().__post_init__()
        first_kernel_size = self.first_kernel_size
        if first_kernel_size and not all(fks % 2 for fks in first_kernel_size):
            raise EvenKernelError(kernel_size=first_kernel_size)

    def get_config_name(self) -> str:
        """
        Return the configuration name based on its attributes.

        Returns
        -------
        str
            A string representation of the configuration.

        """
        name_parts = [
            super().get_config_name(),
        ]
        if self.with_attention:
            name_parts.append('attention')
        if self.without_reducing_filters:
            name_parts.append('without_reducing_filters')
            if self.is_partial_reducing:
                name_parts.append('partial_reducing')
        if self.first_blocks_without_dropout:
            name_parts.append(
                f'{self.first_blocks_without_dropout}without_drop',
            )
        if self.out_residual_blocks_number:
            name_parts.append(
                f'out_res{self.out_residual_blocks_number}',
            )
            if self.is_skipped_with_concat:
                name_parts.append('concat')
            else:
                name_parts.append('sum')
        if self.with_variable_kernel:
            name_parts.append('with_variable_kernel')
        first_kernel_size = self.first_kernel_size
        if first_kernel_size:
            name_parts.append(
                f'first_kernel{first_kernel_size[0]}x{first_kernel_size[1]}',
            )
        if self.vector_length:
            name_parts.append(
                f'vector_length{self.vector_length}',
            )
        return '_'.join(name_parts)


class UNet(BaseCNN):
    """
    A U-Net based segmentation model with configurable architecture.

    Attributes
    ----------
    powers : np.ndarray
        Array representing the depth levels of the network.
    max_pools : list of tf.keras.layers.MaxPooling2D
        Nax pooling layers for down-sampling.
    encoder_layers : list of ConvolutionalBlock
        Encoder convolutional blocks.
    decoder_layers : list of ConvolutionalBlock
        Decoder convolutional blocks.
    decoder_bn_layers : list of tf.keras.layers.BatchNormalization
       Batch normalization layers for the decoder.
    conv_transpose_layers : list of tf.keras.layers.Conv2DTranspose
        Transposed convolution layers for up-sampling.
    attention_gates : list of layers.AttentionGate
        Attention gate layers (if enabled).
    gating_layers : list of layers.GatingSignal
        Gating layers for attention mechanisms (if enabled).
    middle_pair : ConvolutionalBlock
        The bottleneck layer between encoder and decoder.
    output_skipped_connections : layers.OutputSkippedConnections
        The output block with skipped connection.
    flatten_layer : tf.keras.layers.Flatten, optional
        Flattening layer for binary classification (if enabled).
    out_layer : tf.keras.layers.Layer
        Output layer for segmentation or classification.

    """

    config_type = UNetConfig

    def __init__(self, **kwargs) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__(**kwargs)
        self.powers = np.arange(self.config.path_length)
        self.max_pools = []
        self.encoder_layers = []
        self.decoder_layers = []
        self.decoder_bn_layers = []
        self.conv_transpose_layers = []
        self.attention_gates = []
        self.gating_layers = []
        pooling = self.config.pooling
        conv2d_config = self.config.conv_block_config.conv2d_config
        for power in self.powers:
            self.max_pools.append(
                tf.keras.layers.MaxPooling2D(
                    pool_size=(pooling, pooling),
                    padding=conv2d_config.padding,
                ),
            )
            is_dropout_off = False
            if power < self.config.first_blocks_without_dropout:
                is_dropout_off = True
            if self.config.first_kernel_size is not None and not power:
                encoder_kernel_size = self.config.first_kernel_size
            elif self.config.with_variable_kernel:
                ksize = self.config.conv_block_config.conv2d_config.kernel_size
                encoder_kernel_size = tuple(
                    max(
                        4 + kernel_linear_size - 2 * power,
                        kernel_linear_size,
                    )
                    for kernel_linear_size in ksize
                )
            else:
                encoder_kernel_size = None
            self.encoder_layers.append(
                self.get_convolutional_block(
                    filter_scale=pooling ** power,
                    kernel_size=encoder_kernel_size,
                    is_dropout_off=is_dropout_off,
                ),
            )
            filter_scale = pooling ** (self.powers[-1] - power)
            if self.config.with_variable_kernel:
                ksize = self.config.conv_block_config.conv2d_config.kernel_size
                decoder_kernel_size = []
                for kernel_linear_size in ksize:
                    decoder_kernel_linear_size = max(
                        4 + kernel_linear_size - 2 * (self.powers[-1] - power),
                        kernel_linear_size,
                    )
                    decoder_kernel_size.append(decoder_kernel_linear_size)
                decoder_kernel_size = tuple(decoder_kernel_size)
            else:
                decoder_kernel_size = None
            self.decoder_layers.append(
                self.get_convolutional_block(
                    filter_scale=filter_scale,
                    kernel_size=decoder_kernel_size,
                ),
            )
            if self.config.with_attention:
                self.attention_gates.append(
                    self._get_attention_gate(filter_scale=filter_scale),
                )
                self.gating_layers.append(
                    self._get_gating_layer(filter_scale=filter_scale),
                )
            if self.config.conv_block_config.with_bn:
                self.decoder_bn_layers.append(
                    tf.keras.layers.BatchNormalization(),
                )
            self.conv_transpose_layers.append(
                self._get_conv_transpose_layer(filter_scale=filter_scale),
            )
        self.middle_pair = self.get_convolutional_block(
            filter_scale=pooling ** self.powers.size,
        )
        self.output_skipped_connections = cl.SkippedConnections(
            filters=self.config.initial_filters_number,
            config=self.config.conv_block_config,
            is_skipped_with_concat=self.config.is_skipped_with_concat,
            blocks_number=self.config.out_residual_blocks_number,
        )
        if self.config.include_top:
            if self.config.is_binary_classification:
                self.flatten_layer = tf.keras.layers.Flatten()
                self.out_layer = tf.keras.layers.Dense(
                    units=1,
                    activation='sigmoid',
                )
            elif self.config.vector_length:
                self.out_layer = tf.keras.layers.Conv1D(
                    filters=1,
                    kernel_size=self.config.vector_length,
                )
            else:
                self.out_layer = tf.keras.layers.Conv2D(
                    filters=1,
                    kernel_size=conv2d_config.kernel_size,
                    padding=conv2d_config.padding,
                    activation='sigmoid',
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
        outs = []

        out = inputs
        for enc_power in self.powers:
            out = self.encoder_layers[enc_power](out)
            outs.append(out)
            out = self.max_pools[enc_power](outs[-1])

        out = self.middle_pair(out)

        for dec_power in self.powers:
            skipped_connection = outs[self.powers[-1] - dec_power]
            transposed = self.conv_transpose_layers[dec_power](out)
            if self.config.with_attention:
                out = self.gating_layers[dec_power](out)
                out = self.attention_gates[dec_power](
                    [skipped_connection, out],
                )
                skipped_connection = tf.keras.layers.multiply(
                    [skipped_connection, out],
                )
            out = tf.concat([skipped_connection, transposed], axis=-1)
            out = self.decoder_layers[dec_power](out)
            if self.config.conv_block_config.with_bn:
                out = self.decoder_bn_layers[dec_power](out)

        out = self.output_skipped_connections(out)
        if self.config.include_top:
            if self.config.is_binary_classification:
                out = self.flatten_layer(out)
                out = self.out_layer(out)
            elif self.config.vector_length:
                out = tf.image.resize(
                    out,
                    size=(tf.shape(out)[1], self.config.vector_length),
                    method=tf.image.ResizeMethod.BILINEAR,
                )
                out = self.out_layer(out)
            else:
                out = self.out_layer(out)
        return out

    def _get_conv_transpose_layer(
        self,
        filter_scale: int,
    ) -> tf.keras.layers.Conv2DTranspose:
        """
        Return 2D transposed convolution layer.

        Parameters
        ----------
        filter_scale : int
            Scale factor for the number of filters in the convolutional layers.

        Returns
        -------
        tf.keras.layers.Conv2DTranspose
            2D transposed convolution layer.

        """
        filters = self.config.initial_filters_number
        if self.config.is_partial_reducing:
            filters *= filter_scale + self.config.without_reducing_filters
        else:
            filters *= filter_scale * (
                2 ** self.config.without_reducing_filters
            )
        if self.config.max_filters_number:
            filters = min(self.config.max_filters_number, filters)
        return tf.keras.layers.Conv2DTranspose(
            filters=filters,
            strides=(self.config.pooling, self.config.pooling),
            activation=self.config.conv_block_config.activation,
            **self.config.conv_block_config.conv2d_config.as_dict(),
        )

    def _get_attention_gate(
        self,
        filter_scale: int,
    ) -> cl.AttentionGatingBlock:
        """
        Return Attention Gating Block.

        Parameters
        ----------
        filter_scale : int
            Scale factor for the number of filters.

        Returns
        -------
        cl.AttentionGatingBlock
            Attention Gating Block.

        """
        filters = self.config.initial_filters_number
        if self.config.is_partial_reducing:
            filters *= filter_scale + self.config.without_reducing_filters
        else:
            filters *= filter_scale * (
                2 ** self.config.without_reducing_filters
            )
        if self.config.max_filters_number:
            filters = min(self.config.max_filters_number, filters)
        return cl.AttentionGatingBlock(
            filters=filters,
            activation=self.config.conv_block_config.activation,
        )

    def _get_gating_layer(
        self,
        filter_scale: int,
    ) -> cl.GatingSignal:
        """
        Return Gating signal layer for attention mechanism.

        Parameters
        ----------
        filter_scale : int
            Scale factor for the number of filters.

        Returns
        -------
        cl.GatingSignal
            Gating signal layer for attention mechanism.

        """
        filters = self.config.initial_filters_number
        if self.config.is_partial_reducing:
            filters *= filter_scale + self.config.without_reducing_filters
        else:
            filters *= filter_scale * (
                2 ** self.config.without_reducing_filters
            )
        if self.config.max_filters_number:
            filters = min(self.config.max_filters_number, filters)
        return cl.GatingSignal(
            filters=filters,
            activation=self.config.conv_block_config.activation,
            with_bn=self.config.conv_block_config.with_bn,
        )
