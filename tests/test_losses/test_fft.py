import numpy as np
import pytest
import tensorflow as tf

from tf_extensions.losses.fft import FFTLoss, FFTLossConfig

filter_sizes = [11, 22, 33]
losses = [
    'mse',
    'mae',
    'ssim',
]
is_averaged_values = [
    True,
    False,
]
dtypes = [
    'float32',
    'float64',
]
tensor_shapes = [
    (5, 64, 128, 3),
]


default_config = {
    'reduction': 'none',
    'dtype': 'float32',
    'cls_name': 'FFTLoss',
    'is_normalized': False,
    'name': 'fft',
    'loss': 'mse',
    'is_averaged_loss': False,
    'filter_size': 11,
}
non_default_config = {
    'dtype': 'float64',
    'cls_name': 'FFTLoss',
    'is_normalized': True,
    'loss': 'mae',
    'is_averaged_loss': True,
    'filter_size': 13,
}


class TestFFTLossConfig:
    """Class for the FFTLossConfig tests."""

    def test_post_init(self) -> None:
        with pytest.raises(ValueError, match='Unsupported dtype in FFTLoss:'):
            FFTLossConfig(dtype='float16')


class TestFFTLoss:
    """Class for the FFTLoss tests."""

    def test_default_init(self) -> None:
        loss = FFTLoss()
        assert loss.config.name == 'fft'
        assert loss.config.loss == 'mse'
        assert not loss.config.is_averaged_loss
        assert loss.config.filter_size == 11

    def test_default_get_config(self) -> None:
        loss = FFTLoss()
        assert loss.get_config() == default_config

    def test_default_from_config(self) -> None:
        loss = FFTLoss.from_config(non_default_config)
        for attr_name, attr_value in loss.get_config().items():
            if attr_name in non_default_config:
                assert attr_value == non_default_config[attr_name]
            else:
                assert attr_value == default_config[attr_name]

    @pytest.mark.parametrize(
        ('loss', 'filter_size', 'is_averaged_loss'),
        [
            (loss, filter_size, is_averaged_loss)
            for loss in losses
            for filter_size in filter_sizes
            for is_averaged_loss in is_averaged_values
        ],
    )
    def test_init(
        self,
        loss: str,
        filter_size: int,
        is_averaged_loss: bool,
    ) -> None:
        vgg_loss = FFTLoss(
            loss=loss,
            filter_size=filter_size,
            is_averaged_loss=is_averaged_loss,
        )
        assert vgg_loss.config.loss == loss
        assert vgg_loss.config.filter_size == filter_size
        assert vgg_loss.config.is_averaged_loss == is_averaged_loss

    @pytest.mark.parametrize(
        ('loss', 'filter_size', 'is_averaged_loss', 'dtype'),
        [
            (loss, filter_size, is_averaged_loss, dtype)
            for loss in losses
            for filter_size in filter_sizes
            for is_averaged_loss in is_averaged_values
            for dtype in dtypes
        ],
    )
    def test_dtype(
        self,
        loss: str,
        filter_size: int,
        is_averaged_loss: bool,
        dtype: str,
    ) -> None:
        vgg_loss = FFTLoss(
            loss=loss,
            filter_size=filter_size,
            is_averaged_loss=is_averaged_loss,
            dtype=dtype,
        )
        loss = vgg_loss(
            tf.random.normal(shape=tensor_shapes[0]),
            tf.random.normal(shape=tensor_shapes[0]),
        )
        assert loss.dtype.name == dtype

    @pytest.mark.parametrize(
        ('loss', 'filter_size', 'is_averaged_loss', 'shape'),
        [
            (loss, filter_size, is_averaged_loss, shape)
            for loss in losses
            for filter_size in filter_sizes
            for is_averaged_loss in is_averaged_values
            for shape in tensor_shapes
        ],
    )
    def test_shape(
        self,
        loss: str,
        filter_size: int,
        is_averaged_loss: bool,
        shape: tuple[int, ...],
    ) -> None:
        vgg_loss = FFTLoss(
            loss=loss,
            filter_size=filter_size,
            is_averaged_loss=is_averaged_loss,
        )
        loss = vgg_loss(
            tf.random.normal(shape=shape),
            tf.random.normal(shape=shape),
        )
        assert loss.shape == (shape[0], )

    @pytest.mark.parametrize(
        ('loss', 'filter_size', 'is_averaged_loss', 'shape'),
        [
            (loss, filter_size, is_averaged_loss, shape)
            for loss in losses
            for filter_size in filter_sizes
            for is_averaged_loss in is_averaged_values
            for shape in tensor_shapes
        ],
    )
    def test_min_loss(
        self,
        loss: str,
        filter_size: int,
        is_averaged_loss: bool,
        shape: tuple[int, ...],
    ) -> None:
        vgg_loss = FFTLoss(
            loss=loss,
            filter_size=filter_size,
            is_averaged_loss=is_averaged_loss,
        )
        y_true = tf.random.normal(shape=shape)
        loss = vgg_loss(y_true, y_true)
        assert np.allclose(loss.numpy(), np.zeros(shape[0]))

    def test_invalid_loss(self) -> None:
        fft_loss = FFTLoss(
            loss='invalid_loss',
        )
        with pytest.raises(
            ValueError,
            match='Unsupported loss function',
        ):
            fft_loss(
                tf.random.normal(shape=tensor_shapes[0]),
                tf.random.normal(shape=tensor_shapes[0]),
            )

    @pytest.mark.parametrize(
        ('filter_size', 'shape'),
        [
            (34, (5, 64, 128, 3)),
            (66, (5, 128, 128, 3)),
        ],
    )
    def test_invalid_layers_filter_pair(
        self,
        filter_size: int,
        shape: tuple[int, ...],
    ) -> None:
        vgg_loss = FFTLoss(
            loss='ssim',
            filter_size=filter_size,
        )
        with pytest.raises(
            ValueError,
            match=r'Too small image for filter size \d+.',
        ):
            vgg_loss(
                tf.random.normal(shape=shape),
                tf.random.normal(shape=shape),
            )
