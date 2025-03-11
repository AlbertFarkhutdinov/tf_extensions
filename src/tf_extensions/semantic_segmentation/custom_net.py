"""The module provides a custom segmentation network."""
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import display

from tf_extensions.semantic_segmentation import configs as cfg
from tf_extensions.semantic_segmentation.custom_fitter import CustomFitter
from tf_extensions.semantic_segmentation.custom_layers import \
    ConvolutionalBlock


class CustomSegmentationNet(tf.keras.Model):
    """
    A custom segmentation network.

    Parameters
    ----------
    config : cfg.CustomNetConfigType, optional
        Configuration of the network.
        If None, a default configuration is used.
    include_top : bool, optional, default: True
        Whether to include the final classification layer.

    """

    def __init__(
        self,
        config: cfg.CustomNetConfigType = None,
        include_top: bool = True,
    ) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__()
        self.include_top = include_top
        self.config = config or cfg.CustomSegmentationNetConfig()
        kernel_size = self.config.conv_block_config.conv2d_config.kernel_size
        if not kernel_size[0] % 2 or not kernel_size[1] % 2:
            raise ValueError('Odd `kernel_size` is recommended.')

    def call(
        self,
        inputs: tf.Tensor,
        training: Union[bool, tf.Tensor] = None,
        mask: Union[Optional[tf.Tensor], list[Optional[tf.Tensor]]] = None,
    ) -> tf.Tensor:
        """
        Forward pass of the network.

        Parameters
        ----------
        inputs : tf.Tensor
            The input tensor.
        training : bool or tf.Tensor, optional
            Whether the model is in training mode.
        mask : tf.Tensor or list of tf.Tensor, optional
            Mask tensor for specific layers.

        Returns
        -------
        tf.Tensor
            Output tensor.

        """
        return inputs

    def build_graph(self, input_shape: tuple[int, int, int]) -> tf.keras.Model:
        """
        Build a Keras model graph based on the specified input shape.

        Parameters
        ----------
        input_shape : tuple of int
            Shape of the input tensor (height, width, channels).

        Returns
        -------
        tf.keras.Model
            A Keras model instance.

        """
        input_layer = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(
            inputs=[input_layer],
            outputs=self.call(input_layer),
        )

    def plot(
        self,
        input_shape: tuple[int, int, int],
        *args,
        **kwargs,
    ) -> display.Image:
        """
        Plot the model architecture.

        Parameters
        ----------
        input_shape : tuple of int
            Shape of the input tensor (height, width, channels).

        Returns
        -------
        display.Image
            Representation of the model architecture.

        """
        return tf.keras.utils.plot_model(
            self.build_graph(input_shape),
            *args,
            show_shapes=True,
            to_file='{0}.png'.format(self.__class__.__name__),
            **kwargs,
        )

    def summary(self, *args, **kwargs) -> None:
        """Print the summary of the model architecture."""
        return self.build_graph(   # pragma: no cover
            input_shape=kwargs.pop('input_shape'),
        ).summary(*args, **kwargs)

    def get_convolutional_block(
        self,
        filter_scale: int,
        kernel_size: tuple[int, int] = None,
        is_dropout_off: bool = False,
    ) -> ConvolutionalBlock:
        """
        Return a convolutional block with the specified configuration.

        Parameters
        ----------
        filter_scale : int
            Scale factor for the number of filters in the convolutional layers.
        kernel_size : tuple of int, optional
            Kernel size for the convolutional layers.
            If None, the default is used.
        is_dropout_off : bool, optional, default: False
            Whether to disable dropout in the convolutional block.

        Returns
        -------
        ConvolutionalBlock
            A configured convolutional block instance.

        """
        config = deepcopy(self.config.conv_block_config)
        if kernel_size is not None:
            config.conv2d_config.kernel_size = kernel_size
        if is_dropout_off:
            config.with_dropout = False
        filters = self.config.initial_filters_number * filter_scale
        if self.config.max_filters_number:
            filters = min(self.config.max_filters_number, filters)
        return ConvolutionalBlock(
            filters=filters,
            config=config,
        )

    def fit_custom(
        self,
        *dirs,
        samples: np.ndarray,
        labels: np.ndarray,
        with_early_stopping: bool = False,
        loaded_epoch: int = None,
        **fitting_kwargs,
    ) -> tuple[tf.keras.Model, pd.DataFrame]:
        """
        Train the segmentation model or load the previously trained one.

        Parameters
        ----------
        dirs :
            Directories related to model training (e.g., checkpoints, logs).
        samples : np.ndarray
            Training samples.
        labels : np.ndarray
            Corresponding labels for training samples.
        with_early_stopping : bool, optional, default: False
            Whether to use early stopping during training.
        loaded_epoch : int, optional
            The epoch number of a previously saved model to load it.

        Returns
        -------
        tuple
            - The trained model : tf.keras.Model
            - The training history : pd.DataFrame

        """
        fitter = CustomFitter(*dirs, model=self)
        loaded_model, fitting_history = fitter.load_model(
            loaded_epoch=loaded_epoch,
        )
        if loaded_model is None:
            model, fitting_history = fitter.fit_model(
                model=self,
                samples=samples,
                labels=labels,
                with_early_stopping=with_early_stopping,
                **fitting_kwargs,
            )
        else:
            model = loaded_model
        return model, fitting_history
