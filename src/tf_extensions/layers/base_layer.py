"""
Module defining a base TensorFlow layer class and its configuration.

Classes
-------
BaseLayerConfig
    A dataclass for configuring `BaseLayer`.

BaseLayer
    A base class for TensorFlow layers with configuration handling.

"""
from dataclasses import dataclass
from typing import TypeVar

import tensorflow as tf

from tf_extensions.auxiliary.base_config import BaseConfig


@dataclass
class BaseLayerConfig(BaseConfig):
    """
    Configuration for `BaseLayer`.

    Attributes
    ----------
    name : str, optional
        The name of the layer. Defaults to None.

    """

    name: str = None


BaseLayerInstance = TypeVar('BaseLayerInstance', bound='BaseLayer')


class BaseLayer(tf.keras.layers.Layer):
    """
    Base class for TensorFlow layers with configuration management.

    This class serves as a foundation for creating custom TensorFlow layers
    with automatically handling configuration.

    Attributes
    ----------
    config : BaseLayerConfig
        Configuration for `BaseLayer`.

    """

    config_type = BaseLayerConfig

    def __init__(self, **kwargs) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        self.config = self.__class__.config_type.from_kwargs(**kwargs)
        super().__init__()
