"""The module provides a custom Tensorflow model."""
import re
from dataclasses import dataclass

import tensorflow as tf
from IPython import display

from tf_extensions.auxiliary.base_config import BaseConfig
from tf_extensions.auxiliary.custom_types import MaskType, TrainingType


@dataclass
class BaseNetConfig(BaseConfig):
    """
    Configuration of a custom network.

    Attributes
    ----------
    name : str
        Name of the neural network.
    include_top : bool, optional
        Whether to include the top layers in the model. Default is True.

    """

    name: str = ''
    include_top: bool = True

    def __post_init__(self) -> None:
        """Update configuration fields after initialization."""
        if not self.name:
            name = re.sub(
                pattern='([a-z0-9])([A-Z])',
                repl=r'\1_\2',
                string=self.__class__.__name__.removesuffix('Config'),
            )
            name = re.sub(
                pattern='([A-Z]+)([A-Z][a-z])',
                repl=r'\1_\2',
                string=name,
            )
            self.name = name.lower()  # noqa: WPS601

    def get_config_name(self) -> str:
        """
        Return the configuration name based on its attributes.

        Returns
        -------
        str
            A string representation of the configuration.

        """
        if self.include_top:
            return self.name
        return f'{self.name}_without_top'


class BaseNet(tf.keras.Model):
    """
    A custom Tensorflow model.

    Attributes
    ----------
    config : BaseNetConfig
        Configuration of the model.

    """

    config_type = BaseNetConfig

    def __init__(self, **kwargs) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        self.config = self.__class__.config_type.from_kwargs(**kwargs)
        super().__init__()

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
        return inputs

    def build_graph(self, input_shape: tuple[int, ...]) -> tf.keras.Model:
        """
        Build a Keras model graph based on the specified input shape.

        Parameters
        ----------
        input_shape : tuple of int
            The shape of the input tensor, excluding the batch dimension.

        Returns
        -------
        tf.keras.Model
            A compiled Keras model.

        """
        input_layer = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(
            inputs=[input_layer],
            outputs=self.call(input_layer),
        )

    def plot(
        self,
        input_shape: tuple[int, ...],
        *args,
        **kwargs,
    ) -> display.Image:
        """
        Generate a visual representation of the model architecture.

        Parameters
        ----------
        input_shape : tuple of int
            The shape of the input tensor, excluding the batch dimension.

        Returns
        -------
        display.Image
            Representation of the model architecture.

        """
        class_name = self.__class__.__name__
        return tf.keras.utils.plot_model(
            self.build_graph(input_shape),
            *args,
            show_shapes=True,
            to_file=f'{class_name}.png',
            **kwargs,
        )

    def summary(self, *args, **kwargs) -> None:
        """Print the summary of the model architecture."""
        self.build_graph(   # pragma: no cover
            input_shape=kwargs.pop('input_shape'),
        ).summary(*args, **kwargs)
