import pytest
import tensorflow as tf

from tf_extensions import custom_losses as cl
from tf_extensions.custom_losses.ms_dssim import MultiScaleDSSIMConfig

default_config = {
    'reduction': 'none',
    'dtype': 'float32',
    'cls_name': 'MultiScaleDSSIM',
    'is_normalized': False,
    'name': 'ms_dssim',
    'max_val': 2,
    'filter_size': 5,
    'filter_sigma': 1.5,
    'k1': 0.01,
    'k2': 0.03,
    'power_factors': [0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
    'level': 5,
    'with_batches_averaging': False,
    'pool_strides': 2,
    'pool_kernel': 2,
}
non_default_config = {
    'dtype': 'float64',
    'cls_name': 'MultiScaleDSSIM',
    'is_normalized': True,
}


class TestMultiScaleDSSIMConfig:
    """Class for the MultiScaleDSSIMConfig tests."""

    def test_post_init(self) -> None:
        with pytest.raises(
            ValueError,
            match='Level greater than 5 is not supported.',
        ):
            MultiScaleDSSIMConfig(level=6)


class TestMultiScaleSSIM:
    """Class for the MultiScaleSSIM tests."""

    def test_default_init(self) -> None:
        loss = cl.MultiScaleDSSIM()
        assert loss.config.name == 'ms_dssim'
        assert loss.config.max_val == 2
        assert loss.config.filter_size == 5
        assert loss.config.filter_sigma == 1.5
        assert loss.config.k1 == 0.01
        assert loss.config.k2 == 0.03
        assert loss.config.power_factors == [
            0.0448,
            0.2856,
            0.3001,
            0.2363,
            0.1333,
        ]
        assert loss.config.level == 5
        assert not loss.config.with_batches_averaging
        assert loss.config.pool_strides == 2
        assert loss.config.pool_kernel == 2

    def test_default_get_config(self) -> None:
        loss = cl.MultiScaleDSSIM()
        assert loss.get_config() == default_config

    def test_default_from_config(self) -> None:
        loss = cl.MultiScaleDSSIM.from_config(non_default_config)
        for attr_name, attr_value in loss.get_config().items():
            if attr_name in non_default_config:
                assert attr_value == non_default_config[attr_name]
            else:
                assert attr_value == default_config[attr_name]

    @pytest.mark.parametrize(
        ('dtype', 'max_val'),
        [
            ('float16', 2),
            ('float32', 2),
            ('float64', 1),
        ],
    )
    def test_dtype(
        self,
        dtype: str,
        max_val: int,
    ) -> None:
        shape = (5, 128, 128, 3)
        loss = cl.MultiScaleDSSIM(
            dtype=dtype,
            max_val=max_val,
        )(
            tf.random.uniform(shape=shape, minval=-1, maxval=1),
            tf.random.uniform(shape=shape, minval=-1, maxval=1),
        )
        assert loss.dtype.name == dtype

    @pytest.mark.parametrize(
        (
            'shape',
            'with_batches_averaging',
            'max_val',
            'expected_shape',
        ),
        [
            ((3, 128, 128, 7), False, 2, (3, )),
            ((3, 128, 128, 7), True, 2, ()),
            ((3, 128, 128, 7), False, 1, (3, )),
            ((3, 128, 128, 7), True, 1, ()),
        ],
    )
    def test_shape(
        self,
        shape: tuple[int, ...],
        with_batches_averaging: bool,
        max_val: int,
        expected_shape: tuple[int, ...],
    ) -> None:
        loss = cl.MultiScaleDSSIM(
            max_val=max_val,
            with_batches_averaging=with_batches_averaging,
        )(
            tf.random.uniform(shape=shape, minval=-1, maxval=1),
            tf.random.uniform(shape=shape, minval=-1, maxval=1),
        )
        assert loss.shape == expected_shape

    @pytest.mark.parametrize(
        (
            'shape',
            'filter_size',
        ),
        [
            ((3, 128, 128, 7), 11),
        ],
    )
    def test_true_less_min_shape(
        self,
        shape: tuple[int, ...],
        filter_size: int,
    ) -> None:
        loss = cl.MultiScaleDSSIM(
            max_val=2,
            filter_size=filter_size,
        )
        with pytest.raises(
            ValueError,
            match=r'True image \(\d*, \d*\) is less than \(\d*, \d*\)',
        ):
            loss(
                tf.random.uniform(shape=shape, minval=-1, maxval=1),
                tf.random.uniform(shape=shape, minval=-1, maxval=1),
            )

    @pytest.mark.parametrize(
        (
            'true_shape',
            'pred_shape',
            'filter_size',
        ),
        [
            ((3, 256, 256, 7), (3, 128, 128, 7), 11),
        ],
    )
    def test_pred_less_min_shape(
        self,
        true_shape: tuple[int, ...],
        pred_shape: tuple[int, ...],
        filter_size: int,
    ) -> None:
        loss = cl.MultiScaleDSSIM(
            max_val=2,
            filter_size=filter_size,
        )
        with pytest.raises(
            ValueError,
            match=r'Predicted image \(\d*, \d*\) is less than \(\d*, \d*\)',
        ):
            loss(
                y_true=tf.random.uniform(
                    shape=true_shape,
                    minval=-1,
                    maxval=1,
                ),
                y_pred=tf.random.uniform(
                    shape=pred_shape,
                    minval=-1,
                    maxval=1,
                ),
            )
