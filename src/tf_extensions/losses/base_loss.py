"""
Module providing a base class for custom loss functions in TensorFlow/Keras.

Classes
-------
BaseLossConfig
    Configuration class for `BaseLoss`.
BaseLoss
    Base class for implementing custom TensorFlow/Keras loss functions.

"""
from dataclasses import dataclass
from typing import Any, Type, TypeVar, Union

import tensorflow as tf
from keras import backend as kb

from tf_extensions.auxiliary.base_config import BaseConfig


@dataclass
class BaseLossConfig(BaseConfig):
    """
    Configuration class for `BaseLoss`.

    Attributes
    ----------
    reduction : tf.keras.losses.Reduction, default: `Reduction.NONE`
        A reduction method for computing loss.
    name : str, optional
        Name of the loss function.
    dtype : str, default: 'float32'
        Data type of tf.Tensor.
    is_normalized : bool
        Whether to normalize input images before computing loss.

    """

    reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.NONE
    name: str = None
    dtype: str = 'float32'
    is_normalized: bool = False


BaseLossInstance = TypeVar('BaseLossInstance', bound='BaseLoss')


class BaseLoss(tf.keras.losses.Loss):
    """
    Base class for custom loss functions in TensorFlow/Keras.

    Attributes
    ----------
    config : BaseLossConfig
        Configuration settings for the loss function.

    """

    config_type = BaseLossConfig

    def __init__(self, **kwargs) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        self.config = self.__class__.config_type.from_kwargs(**kwargs)
        super().__init__(
            name=self.config.name,
            reduction=self.config.reduction,
        )

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """
        Return the loss values.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth values, shape [batch_size, d0, ..., dN].
        y_pred : tf.Tensor
            Predicted values, shape [batch_size, d0, ..., dN].

        Raises
        ------
        NotImplementedError
            If this method is not implemented.

        """
        raise NotImplementedError()  # pragma: no cover

    def get_config(self) -> dict[str, Any]:  # noqa: WPS615
        """
        Return the configuration dictionary for serialization.

        Returns
        -------
        dict
            Configuration dictionary containing loss parameters.

        """
        config = super().get_config()
        config['cls_name'] = self.__class__.__name__
        config.update(self.config.as_dict())
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, tf.keras.losses.Loss):
                config[attr_name] = attr_value.get_config()
        return config

    def cast_to_dtype(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Cast input tensors to the required data type.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth values.
        y_pred : tf.Tensor
            Predicted values.

        Returns
        -------
        tuple of tf.Tensor
            Tensors cast to the required data type.

        """
        y_true = tf.cast(y_true, dtype=self.config.dtype)
        y_pred = tf.cast(y_pred, dtype=self.config.dtype)
        return y_true, y_pred

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BaseLossInstance:
        """
        Create a loss instance from a configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        BaseLossInstance
            New instance of `BaseLoss`.

        """
        loss_config = {}
        for attr_name, attr_value in config.items():
            try:
                loss_config[attr_name] = attr_value.from_config()
            except AttributeError:
                loss_config[attr_name] = attr_value
        return cls(**loss_config)

    def get_loss_attribute(
        self,
        config: Union[BaseLossInstance, dict[str, Any], None],
        loss_cls: Type[BaseLossInstance],
    ) -> BaseLossInstance:
        """
        Return or create a loss instance from the given configuration.

        Parameters
        ----------
        config : BaseLossInstance or dict, optional
            Loss instance or configuration dictionary.
        loss_cls : Type[BaseLossInstance]
            Loss class type.

        Returns
        -------
        BaseLossInstance
            Initialized loss instance.

        """
        if isinstance(config, dict):
            return loss_cls.from_config(config)
        if config is not None:
            return config
        return loss_cls(dtype=self.config.dtype)

    def normalize_images(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Normalize images before loss computation if required.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth values.
        y_pred : tf.Tensor
            Predicted values.

        Returns
        -------
        tuple of tf.Tensor
            Normalized tensors.

        """
        if not self.config.is_normalized:
            return y_true, y_pred
        mean_axis = tf.range(
            start=1,
            limit=kb.ndim(y_true),
        )
        batch_scales = kb.max(
            tf.abs(y_true),
            axis=mean_axis,
        )
        norm_coefficient = kb.switch(
            batch_scales != 0,
            batch_scales,
            kb.ones_like(batch_scales),
        )
        shape = (-1, 1, 1, 1)
        norm_coefficient = tf.reshape(
            tensor=norm_coefficient,
            shape=shape,
        )
        return y_true / norm_coefficient, y_pred / norm_coefficient
