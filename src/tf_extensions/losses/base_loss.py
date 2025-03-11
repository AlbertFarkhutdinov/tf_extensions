from dataclasses import dataclass
from typing import Any, Type, TypeVar, Union

import tensorflow as tf
from keras import backend as kb

from tf_extensions.auxiliary.base_config import BaseConfig


@dataclass
class BaseLossConfig(BaseConfig):
    reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.NONE
    name: str = None
    dtype: str = 'float32'
    is_normalized: bool = False


BaseLossInstance = TypeVar('BaseLossInstance', bound='BaseLoss')


class BaseLoss(tf.keras.losses.Loss):
    """Class for the CustomLoss."""

    config_type = BaseLossConfig

    def __init__(self, **kwargs) -> None:
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
        Invoke the `Loss` instance.

        Parameters
        ----------
        y_true : array-like
            Ground truth values with shape `[batch_size, d0, ..., dN]`.
        y_pred : array-like
            Predicted values shape `[batch_size, d0, ..., dN]`.

        Returns
        -------
        tf.Tensor
            The loss function value.

        """
        raise NotImplementedError()  # pragma: no cover

    def get_config(self) -> dict[str, Any]:  # noqa: WPS615
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
        y_true = tf.cast(y_true, dtype=self.config.dtype)
        y_pred = tf.cast(y_pred, dtype=self.config.dtype)
        return y_true, y_pred

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BaseLossInstance:
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
        if not self.config.is_normalized:
            return y_true, y_pred
        ndim = kb.ndim(y_true)
        mean_axis = tf.range(
            start=1,
            limit=ndim,
        )
        mean_abs_true = kb.mean(
            tf.abs(y_true),
            axis=mean_axis,
        )
        norm_coefficient = kb.switch(
            mean_abs_true != 0,
            mean_abs_true,
            kb.ones_like(mean_abs_true),
        )
        shape = (-1, 1, 1, 1)
        norm_coefficient = tf.reshape(
            tensor=norm_coefficient,
            shape=shape,
        )
        return y_true / norm_coefficient, y_pred / norm_coefficient
