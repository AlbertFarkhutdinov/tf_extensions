import numpy as np
import pytest
import tensorflow as tf

import tf_extensions.losses as cl

LOSSES = (
    cl.DISTS(),
    cl.DSSIM(),
    cl.FFTLoss(),
    cl.SoftDiceLoss(),
    cl.VGGLoss(),
)


class TestMultiscaleLoss:
    """Class for the MultiscaleLoss tests."""

    @pytest.mark.parametrize(
        ('base_loss', 'weights', 'name', 'expected_name'),
        [
            (LOSSES[0], None, 'test_name', 'test_name'),
            (LOSSES[1], [0.5, 0.5], None, 'multiscale_dssim'),
            (LOSSES[2], [0.5, 0.5], 'test_name', 'test_name'),
        ],
    )
    def test_multiscale_loss_init(
        self,
        base_loss: tf.keras.losses.Loss,
        weights: list[float],
        name: str,
        expected_name: str,
    ) -> None:
        ms_loss = cl.MultiScaleLoss(
            base_loss=base_loss,
            weights=weights,
            name=name,
        )
        ms_config = ms_loss.config
        assert ms_loss.name == expected_name
        assert ms_config.name == expected_name
        assert type(ms_config.base_loss) is type(base_loss)
        assert ms_config.base_loss.get_config() == base_loss.get_config()

    @pytest.mark.parametrize(
        ('base_loss', 'weights', 'name', 'expected_name'),
        [
            (LOSSES[0], None, 'test_name', 'test_name'),
            (LOSSES[1], [0.5, 0.5], None, 'multiscale_dssim'),
            (LOSSES[2], [0.5, 0.5], 'test_name', 'test_name'),
        ],
    )
    def test_multiscale_loss_get_config(
        self,
        base_loss: tf.keras.losses.Loss,
        weights: list[float],
        name: str,
        expected_name: str,
    ) -> None:
        loss = cl.MultiScaleLoss(
            base_loss=base_loss,
            weights=weights,
            name=name,
        )
        assert loss.get_config() == {
            'reduction': 'none',
            'dtype': 'float32',
            'is_normalized': False,
            'cls_name': 'MultiScaleLoss',
            'name': expected_name,
            'base_loss': base_loss.get_config(),
            'weights': weights,
        }

    @pytest.mark.parametrize(
        ('base_loss', 'weights', 'name', 'expected_name'),
        [
            (LOSSES[0], None, 'test_name', 'test_name'),
            (LOSSES[1], [0.5, 0.5], None, 'multiscale_dssim'),
            (LOSSES[2], [0.5, 0.5], 'test_name', 'test_name'),
        ],
    )
    def test_multiscale_loss_from_config(
        self,
        base_loss: tf.keras.losses.Loss,
        weights: list[float],
        name: str,
        expected_name: str,
    ) -> None:
        config = {
            'reduction': 'none',
            'dtype': 'float64',
            'is_normalized': True,
            'cls_name': 'MultiScaleLoss',
            'name': name,
            'base_loss': base_loss.get_config(),
            'weights': weights,
        }
        loss = cl.MultiScaleLoss.from_config(config)
        for attr_name, attr_value in loss.get_config().items():
            if attr_name in config:
                if attr_name == 'name':
                    assert attr_value == expected_name
                else:
                    assert config[attr_name] == attr_value

    def test_without_base_loss(self) -> None:
        with pytest.raises(
            ValueError,
            match='Loss must be provided.',
        ):
            cl.MultiScaleLoss()

    def test_unsupported_input_types(self) -> None:
        ms_loss = cl.MultiScaleLoss(base_loss=cl.SoftDiceLoss())
        with pytest.raises(
            ValueError,
            match='Inputs must be tuples of tensors.',
        ):
            ms_loss(y_true=5, y_pred=3)

    def test_different_lengths(self) -> None:
        ms_loss = cl.MultiScaleLoss(base_loss=cl.SoftDiceLoss())
        with pytest.raises(
            ValueError,
            match='Lengths of y_true and y_pred must match.',
        ):
            ms_loss(
                y_true=(
                    tf.random.normal(shape=(5, 128, 128, 3)),
                ),
                y_pred=(
                    tf.random.normal(shape=(5, 128, 128, 3)),
                    tf.random.normal(shape=(5, 64, 64, 3)),
                ),
            )

    def test_incorrect_weights_length(self) -> None:
        ms_loss = cl.MultiScaleLoss(
            base_loss=cl.SoftDiceLoss(),
            weights=[1],
        )
        with pytest.raises(
            ValueError,
            match='Lengths of weights and y_true must match.',
        ):
            ms_loss(
                y_true=(
                    tf.random.normal(shape=(5, 128, 128, 3)),
                    tf.random.normal(shape=(5, 64, 64, 3)),
                ),
                y_pred=(
                    tf.random.normal(shape=(5, 128, 128, 3)),
                    tf.random.normal(shape=(5, 64, 64, 3)),
                ),
            )

    def test_incorrect_batch_size_true(self) -> None:
        ms_loss = cl.MultiScaleLoss(base_loss=cl.SoftDiceLoss())
        with pytest.raises(
            ValueError,
            match='Batch sizes in y_true must match.',
        ):
            ms_loss(
                y_true=(
                    tf.random.normal(shape=(5, 128, 128, 3)),
                    tf.random.normal(shape=(3, 64, 64, 3)),
                ),
                y_pred=(
                    tf.random.normal(shape=(5, 128, 128, 3)),
                    tf.random.normal(shape=(5, 64, 64, 3)),
                ),
            )

    def test_incorrect_batch_size_pred(self) -> None:
        ms_loss = cl.MultiScaleLoss(base_loss=cl.SoftDiceLoss())
        with pytest.raises(
            ValueError,
            match='Batch sizes in y_pred must match.',
        ):
            ms_loss(
                y_true=(
                    tf.random.normal(shape=(5, 128, 128, 3)),
                    tf.random.normal(shape=(5, 64, 64, 3)),
                ),
                y_pred=(
                    tf.random.normal(shape=(5, 128, 128, 3)),
                    tf.random.normal(shape=(3, 64, 64, 3)),
                ),
            )

    @pytest.mark.parametrize(
        ('base_loss', 'weights', 'dtype', 'batch_size'),
        [
            (LOSSES[0], [1, 1], 'float64', 5),
            (LOSSES[1], [1, 1], 'float64', 5),
            (LOSSES[2], [1, 1], 'float64', 5),
            (LOSSES[3], [1, 1], 'float64', 5),
            (LOSSES[4], [1, 1], 'float64', 5),
        ],
    )
    def test_dtype_and_shape(
        self,
        base_loss: tf.keras.losses.Loss,
        weights: list[float],
        dtype: str,
        batch_size: int,
    ) -> None:
        y_true = (
            tf.random.normal(shape=(batch_size, 128, 128, 3)),
            tf.random.normal(shape=(batch_size, 64, 64, 3)),
        )
        y_pred = (
            tf.random.normal(shape=(batch_size, 128, 128, 3)),
            tf.random.normal(shape=(batch_size, 64, 64, 3)),
        )
        ms_loss = cl.MultiScaleLoss(
            base_loss=base_loss,
            weights=weights,
            dtype=dtype,
        )(y_true, y_pred)
        assert ms_loss.dtype.name == dtype
        assert ms_loss.shape == (batch_size, )

    @pytest.mark.parametrize(
        ('base_loss', 'weights', 'batch_size'),
        [
            (LOSSES[0], [1, 1], 5),
            (LOSSES[1], [1, 1], 5),
            (LOSSES[2], [1, 1], 5),
            (LOSSES[3], [1, 1], 5),
            (LOSSES[4], [1, 1], 5),
        ],
    )
    def test_min_loss(
        self,
        base_loss: tf.keras.losses.Loss,
        weights: list[float],
        batch_size: int,
    ) -> None:
        y_true = (
            tf.random.normal(shape=(batch_size, 128, 128, 3)),
            tf.random.normal(shape=(batch_size, 64, 64, 3)),
        )
        loss = cl.MultiScaleLoss(
            base_loss=base_loss,
            weights=weights,
        )(y_true, y_true)
        assert np.allclose(loss.numpy(), np.zeros(batch_size))
