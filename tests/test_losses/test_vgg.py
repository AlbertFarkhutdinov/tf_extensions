import numpy as np
import pytest
import tensorflow as tf

from tf_extensions.auxiliary import exceptions
from tf_extensions.losses.vgg import VGGLoss

valid_layers_and_filters = [
    (['block5_conv4'], 3),
    (['block3_conv4'], 11),
    (['block3_conv4', 'block5_conv4'], 3),
]
invalid_layers_and_filters = [
    (['block5_conv4'], 11),
    (['block3_conv4', 'block5_conv4'], 11),
]
losses = [
    'mse',
    'mae',
    'ssim',
]
is_preprocessed_values = [
    True,
    False,
]
dtypes = [
    'float16',
    'float32',
    'float64',
]
tensor_shapes = [
    (5, 64, 128, 3),
]
batch_sizes = [None, 64]


default_config = {
    'reduction': 'none',
    'dtype': 'float32',
    'cls_name': 'VGGLoss',
    'is_normalized': False,
    'name': 'vgg',
    'layer_names': ['block5_conv4'],
    'loss': 'mse',
    'is_preprocessed': True,
    'filter_size': 11,
    'batch_size': None,
}
non_default_config = {
    'dtype': 'float64',
    'cls_name': 'VGGLoss',
    'is_normalized': True,
    'loss': 'mae',
    'is_preprocessed': False,
    'filter_size': 13,
    'batch_size': 64,
}


class TestVGGLoss:
    """Class for the VGGLoss tests."""

    def test_default_init(self) -> None:
        loss = VGGLoss()
        assert loss.config.name == 'vgg'
        assert loss.config.layer_names == ['block5_conv4']
        assert loss.config.loss == 'mse'
        assert loss.config.is_preprocessed
        assert loss.config.filter_size == 11

    def test_default_get_config(self) -> None:
        loss = VGGLoss()
        assert loss.get_config() == default_config

    def test_default_from_config(self) -> None:
        loss = VGGLoss.from_config(non_default_config)
        for attr_name, attr_value in loss.get_config().items():
            if attr_name in non_default_config:
                assert attr_value == non_default_config[attr_name]
            else:
                assert attr_value == default_config[attr_name]

    @pytest.mark.parametrize(
        ('loss', 'is_preprocessed', 'layer_filter_pair'),
        [
            (loss, is_preprocessed, layer_filter_pair)
            for loss in losses
            for is_preprocessed in is_preprocessed_values
            for layer_filter_pair in valid_layers_and_filters
        ],
    )
    def test_init(
        self,
        loss: str,
        is_preprocessed: bool,
        layer_filter_pair: tuple[list[str], int],
    ) -> None:
        vgg_loss = VGGLoss(
            loss=loss,
            is_preprocessed=is_preprocessed,
            layer_names=layer_filter_pair[0],
            filter_size=layer_filter_pair[1],
        )
        assert vgg_loss.config.layer_names == layer_filter_pair[0]
        assert vgg_loss.config.loss == loss
        assert vgg_loss.config.is_preprocessed == is_preprocessed
        assert vgg_loss.config.filter_size == layer_filter_pair[1]

    @pytest.mark.parametrize(
        (
            'loss',
            'is_preprocessed',
            'layer_filter_pair',
            'dtype',
            'batch_size',
        ),
        [
            (loss, is_preprocessed, layer_filter_pair, dtype, batch_size)
            for loss in losses
            for is_preprocessed in is_preprocessed_values
            for layer_filter_pair in valid_layers_and_filters
            for dtype in dtypes
            for batch_size in batch_sizes
        ],
    )
    def test_dtype(
        self,
        loss: str,
        is_preprocessed: bool,
        layer_filter_pair: tuple[list[str], int],
        dtype: str,
        batch_size: int,
    ) -> None:
        vgg_loss = VGGLoss(
            loss=loss,
            is_preprocessed=is_preprocessed,
            layer_names=layer_filter_pair[0],
            filter_size=layer_filter_pair[1],
            dtype=dtype,
            batch_size=batch_size,
        )
        loss = vgg_loss(
            tf.random.normal(shape=tensor_shapes[0]),
            tf.random.normal(shape=tensor_shapes[0]),
        )
        assert loss.dtype.name == dtype

    @pytest.mark.parametrize(
        (
            'loss',
            'is_preprocessed',
            'layer_filter_pair',
            'shape',
            'batch_size',
        ),
        [
            (loss, is_preprocessed, layer_filter_pair, shape, batch_size)
            for loss in losses
            for is_preprocessed in is_preprocessed_values
            for layer_filter_pair in valid_layers_and_filters
            for shape in tensor_shapes
            for batch_size in batch_sizes
        ],
    )
    def test_shape(
        self,
        loss: str,
        is_preprocessed: bool,
        layer_filter_pair: tuple[list[str], int],
        shape: tuple[int, ...],
        batch_size: int,
    ) -> None:
        vgg_loss = VGGLoss(
            loss=loss,
            is_preprocessed=is_preprocessed,
            layer_names=layer_filter_pair[0],
            filter_size=layer_filter_pair[1],
            batch_size=batch_size,
        )
        loss = vgg_loss(
            tf.random.normal(shape=shape),
            tf.random.normal(shape=shape),
        )
        assert loss.shape == (shape[0], )

    @pytest.mark.parametrize(
        (
            'loss',
            'is_preprocessed',
            'layer_filter_pair',
            'shape',
            'batch_size',
        ),
        [
            (loss, is_preprocessed, layer_filter_pair, shape, batch_size)
            for loss in losses
            for is_preprocessed in is_preprocessed_values
            for layer_filter_pair in valid_layers_and_filters
            for shape in tensor_shapes
            for batch_size in batch_sizes
        ],
    )
    def test_min_loss(
        self,
        loss: str,
        is_preprocessed: bool,
        layer_filter_pair: tuple[list[str], int],
        shape: tuple[int, ...],
        batch_size: int,
    ) -> None:
        vgg_loss = VGGLoss(
            loss=loss,
            is_preprocessed=is_preprocessed,
            layer_names=layer_filter_pair[0],
            filter_size=layer_filter_pair[1],
            batch_size=batch_size,
        )
        y_true = tf.random.normal(shape=shape)
        loss = vgg_loss(y_true, y_true)
        assert np.allclose(
            loss.numpy(),
            np.zeros(shape[0]),
            atol=1e-5,
        )

    @pytest.mark.parametrize(
        ('is_preprocessed', 'layer_filter_pair'),
        [
            (is_preprocessed, layer_filter_pair)
            for is_preprocessed in is_preprocessed_values
            for layer_filter_pair in invalid_layers_and_filters
        ],
    )
    def test_invalid_layers_filter_pair(
        self,
        is_preprocessed: bool,
        layer_filter_pair: tuple[list[str], int],
    ) -> None:
        vgg_loss = VGGLoss(
            loss='ssim',
            is_preprocessed=is_preprocessed,
            layer_names=layer_filter_pair[0],
            filter_size=layer_filter_pair[1],
        )
        with pytest.raises(
            exceptions.WrongNumberError,
            match='Too big filter size for the specified VGG layer.',
        ):
            vgg_loss(
                tf.random.normal(shape=tensor_shapes[0]),
                tf.random.normal(shape=tensor_shapes[0]),
            )

    @pytest.mark.parametrize(
        ('loss', 'layer_filter_pair'),
        [
            (loss, layer_filter_pair)
            for loss in losses
            for layer_filter_pair in valid_layers_and_filters
        ],
    )
    def test_invalid_channels(
        self,
        loss: str,
        layer_filter_pair: tuple[list[str], int],
    ) -> None:
        vgg_loss = VGGLoss(
            loss=loss,
            is_preprocessed=True,
            layer_names=layer_filter_pair[0],
            filter_size=layer_filter_pair[1],
        )
        invalid_shape = (3, 128, 256, 1)
        valid_shape = (3, 128, 256, 3)
        with pytest.raises(
            exceptions.NonEqualValuesError,
            match=r'True image has \d+ channels. Required: 3.',
        ):
            vgg_loss(
                y_true=tf.random.normal(shape=invalid_shape),
                y_pred=tf.random.normal(shape=valid_shape),
            )
        with pytest.raises(
            exceptions.NonEqualValuesError,
            match=r'Predicted image has \d+ channels. Required: 3.',
        ):
            vgg_loss(
                y_true=tf.random.normal(shape=valid_shape),
                y_pred=tf.random.normal(shape=invalid_shape),
            )

    def test_invalid_loss(self) -> None:
        vgg_loss = VGGLoss(
            loss='invalid_loss',
            is_preprocessed=True,
            layer_names=valid_layers_and_filters[0][0],
            filter_size=valid_layers_and_filters[0][1],
        )
        with pytest.raises(
            exceptions.UnsupportedArgumentError,
            match='Unsupported value of "loss":',
        ):
            vgg_loss(
                tf.random.normal(shape=tensor_shapes[0]),
                tf.random.normal(shape=tensor_shapes[0]),
            )
