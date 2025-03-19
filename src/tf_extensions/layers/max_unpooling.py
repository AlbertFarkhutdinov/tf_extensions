"""The module contains 2D Max Unpooling layer with Argmax input."""
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from tf_extensions.auxiliary.base_config import BaseConfig
from tf_extensions.layers.base_layer import BaseLayer


@dataclass
class MaxUnPoolingConfig(BaseConfig):
    """
    Config of 2D Max Unpooling layer with Argmax input.

    Parameters
    ----------
    pool_size : tuple of int, optional, default: (2, 2)
        Size of the max pooling window.

    """

    pool_size: tuple[int, ...] = (2, 2)


class MaxUnpooling2D(BaseLayer):
    """
    2D Max Unpooling layer for reconstructing feature maps.

    This layer performs the inverse operation of max pooling
    by placing pooled values back into their original positions
    using the provided indices.

    Attributes
    ----------
    config: MaxUnPoolingConfig
        Config of MaxUnpooling2D.

    """

    config_type = MaxUnPoolingConfig

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

        output_shape = input_shape * np.array([1, *self.config.pool_size, 1])
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
                input_shape[1] * self.config.pool_size[0],
                input_shape[2] * self.config.pool_size[1],
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
            mask_shape[1] * self.config.pool_size[0],
            mask_shape[2] * self.config.pool_size[1],
            mask_shape[3],
        )
