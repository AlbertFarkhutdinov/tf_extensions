import numpy as np
import pytest
import tensorflow as tf

from tf_extensions.custom_losses.dssim import DSSIM

default_config = {
    'reduction': 'none',
    'dtype': 'float32',
    'cls_name': 'DSSIM',
    'is_normalized': False,
    'name': 'dssim',
    'max_val': 2,
    'filter_size': 5,
    'filter_sigma': 1.5,
    'k1': 0.01,
    'k2': 0.03,
    'return_cs_map': False,
    'return_index_map': False,
    'with_channels_averaging': True,
}
non_default_config = {
    'dtype': 'float64',
    'cls_name': 'DSSIM',
    'is_normalized': True,
}


class TestSSIM:
    """Class for the DSSIM tests."""

    def test_default_init(self) -> None:
        loss = DSSIM()
        assert loss.name == 'dssim'
        assert loss.config.name == 'dssim'
        assert loss.config.max_val == 2
        assert loss.config.filter_size == 5
        assert loss.config.filter_sigma == 1.5
        assert loss.config.k1 == 0.01
        assert loss.config.k2 == 0.03
        assert not loss.config.return_cs_map
        assert not loss.config.return_index_map
        assert loss.config.with_channels_averaging

    def test_default_get_config(self) -> None:
        loss = DSSIM()
        assert loss.get_config() == default_config

    def test_default_from_config(self) -> None:
        loss = DSSIM.from_config(non_default_config)
        for attr_name, attr_value in loss.get_config().items():
            if attr_name in non_default_config:
                assert attr_value == non_default_config[attr_name]
            else:
                assert attr_value == default_config[attr_name]

    @pytest.mark.parametrize(
        ('dtype',),
        [
            ('float16',),
            ('float32',),
            ('float64',),
        ],
    )
    def test_dtype(
        self,
        dtype: str,
    ) -> None:
        shape = (5, 128, 128, 3)
        loss = DSSIM(
            dtype=dtype,
            max_val=2,
        )(
            tf.random.uniform(shape=shape, minval=-1, maxval=1),
            tf.random.uniform(shape=shape, minval=-1, maxval=1),
        )
        assert loss.dtype.name == dtype

    @pytest.mark.parametrize(
        (
            'shape',
            'filter_size',
            'return_cs_map',
            'return_index_map',
            'with_channels_averaging',
            'expected_shape',
        ),
        [
            ((3, 128, 128, 7), 11, False, False, False, (3, 7)),
            ((3, 128, 128, 7), 11, False, False, True, (3, )),
            ((3, 128, 128, 7), 11, False, True, False, (3, 118, 118, 7)),
            ((3, 128, 128, 7), 11, False, True, True, (3, 118, 118)),
            ((3, 128, 128, 7), 11, True, False, False, (2, 3, 7)),
            ((3, 128, 128, 7), 11, True, False, True, (2, 3)),
            ((3, 128, 128, 7), 11, True, True, False, (2, 3, 118, 118, 7)),
            ((3, 128, 128, 7), 11, True, True, True, (2, 3, 118, 118)),
        ],
    )
    def test_shape(
        self,
        shape: tuple[int, ...],
        filter_size: int,
        return_cs_map: bool,
        return_index_map: bool,
        with_channels_averaging: bool,
        expected_shape: tuple[int, ...],
    ) -> None:
        loss = DSSIM(
            max_val=2,
            filter_size=filter_size,
            return_cs_map=return_cs_map,
            return_index_map=return_index_map,
            with_channels_averaging=with_channels_averaging,
        )(
            tf.random.uniform(shape=shape, minval=-1, maxval=1),
            tf.random.uniform(shape=shape, minval=-1, maxval=1),
        )
        assert loss.shape == expected_shape

    @pytest.mark.parametrize(
        (
            'shape',
            'filter_size',
            'return_cs_map',
            'return_index_map',
            'with_channels_averaging',
        ),
        [
            ((3, 128, 128, 7), 11, False, True, True),
            ((3, 128, 128, 7), 11, False, False, True),
        ],
    )
    def test_tf_ssim(
        self,
        shape: tuple[int, ...],
        filter_size: int,
        return_cs_map: bool,
        return_index_map: bool,
        with_channels_averaging: bool,
    ) -> None:
        y_true = tf.random.uniform(shape=shape, minval=-1, maxval=1)
        y_pred = tf.random.uniform(shape=shape, minval=-1, maxval=1)
        max_val = 2
        loss1 = DSSIM(
            max_val=max_val,
            filter_size=filter_size,
            return_cs_map=return_cs_map,
            return_index_map=return_index_map,
            with_channels_averaging=with_channels_averaging,
        )(y_true, y_pred)
        loss2 = tf.image.ssim(
            y_true,
            y_pred,
            max_val=max_val,
            filter_size=filter_size,
            return_index_map=return_index_map,
        )
        assert np.allclose(loss1, (1 - loss2) / 2, atol=1e-6)
