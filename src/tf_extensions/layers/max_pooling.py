"""The module contains 2D Max Pooling layer with Argmax output."""
from dataclasses import dataclass
from typing import Union

import tensorflow as tf

from tf_extensions.auxiliary.base_config import BaseConfig
from tf_extensions.layers.base_layer import BaseLayer


@dataclass
class MaxPoolingConfig(BaseConfig):
    """
    Config of 2D Max Pooling layer with Argmax output.

    Parameters
    ----------
    pool_size : tuple of int, optional, default: (2, 2)
        Size of the max pooling window.
    strides : tuple of int, optional, default: (2, 2)
        Stride of the pooling operation.
    padding : str, optional, default: 'same'
        Padding method, either 'same' or 'valid'.

    """

    pool_size: tuple[int, ...] = (2, 2)
    strides: tuple[int, ...] = (2, 2)
    padding: str = 'same'


class MaxPoolingWithArgmax2D(BaseLayer):
    """
    2D Max Pooling layer with Argmax output.

    This layer performs max pooling
    and also returns the indices of the max values,
    which can be useful for operations like unpooling.

    Attributes
    ----------
    config: MaxPoolingConfig
        Config of MaxPoolingWithArgmax2D.

    """

    config_type = MaxPoolingConfig

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
            ksize=[1, *self.config.pool_size[:2], 1],
            strides=[1, *self.config.strides[:2], 1],
            padding=self.config.padding.upper(),
        )
        argmax = tf.cast(argmax, 'int32')
        return output, argmax

    def compute_output_shape(  # noqa: PLR6301
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

    def compute_mask(  # noqa: PLR6301
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
