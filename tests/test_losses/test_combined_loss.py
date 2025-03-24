import numpy as np
import pytest
import tensorflow as tf

import tf_extensions.losses as cl
from tf_extensions.auxiliary import exceptions

COMBINATIONS = (
    [cl.FFTLoss(), cl.FFTLoss()],
    [cl.FFTLoss(), cl.SoftDiceLoss()],
)


class TestCombinedLoss:
    """Class for the CombinedLoss tests."""

    @pytest.mark.parametrize(
        ('losses', 'weights', 'name', 'expected_name'),
        [
            (COMBINATIONS[0], [1, 1], None, 'fft_fft'),
            (COMBINATIONS[1], [1, 1], None, 'fft_sdl'),
        ],
    )
    def test_combined_loss_init(
        self,
        losses: list,
        weights: list[float],
        name: str,
        expected_name: str,
    ) -> None:
        loss = cl.CombinedLoss(
            losses=losses,
            weights=weights,
            name=name,
        )
        config = loss.config
        assert loss.name == expected_name
        assert config.name == expected_name
        assert len(config.losses) == len(losses)
        assert len(config.weights) == len(weights)
        for config_loss, loss in zip(config.losses, losses):
            assert type(config_loss) is type(loss)
            assert config_loss.get_config() == loss.get_config()

    @pytest.mark.parametrize(
        ('losses', 'weights', 'expected_name'),
        [
            (COMBINATIONS[0], [1, 1], 'fft_fft'),
            (COMBINATIONS[1], [1, 1], 'fft_sdl'),
        ],
    )
    def test_combined_loss_get_config(
        self,
        losses: list,
        weights: list[float],
        expected_name: str,
    ) -> None:
        loss = cl.CombinedLoss(
            losses=losses,
            weights=weights,
        )
        assert loss.get_config() == {
            'reduction': 'none',
            'dtype': 'float32',
            'is_normalized': False,
            'cls_name': 'CombinedLoss',
            'name': expected_name,
            'losses': [loss.get_config() for loss in losses],
            'weights': weights,
        }

    @pytest.mark.parametrize(
        ('losses', 'weights'),
        [
            (COMBINATIONS[0], [1, 1]),
            (COMBINATIONS[1], [1, 1]),
        ],
    )
    def test_combined_loss_from_config(
        self,
        losses: list,
        weights: list[float],
    ) -> None:
        config = {
            'reduction': 'none',
            'dtype': 'float64',
            'is_normalized': True,
            'cls_name': 'CombinedLoss',
            'name': 'custom_name',
            'losses': [loss.get_config() for loss in losses],
            'weights': weights,
        }
        loss = cl.CombinedLoss.from_config(config)
        for attr_name, attr_value in loss.get_config().items():
            if attr_name in config:
                assert attr_value == config[attr_name]

    def test_combined_loss_without_losses(self) -> None:
        with pytest.raises(
            exceptions.UnspecifiedArgumentError,
            match='Unspecified argument: "losses".',
        ):
            cl.CombinedLoss()

    def test_combined_loss_without_weights(self) -> None:
        with pytest.raises(
            exceptions.UnspecifiedArgumentError,
            match='Unspecified argument: "weights".',
        ):
            cl.CombinedLoss(losses=[])

    def test_combined_loss_with_different_lengths(self) -> None:
        with pytest.raises(
            exceptions.NonEqualValuesError,
            match='Losses and weights lists must have the same length.',
        ):
            cl.CombinedLoss(losses=[cl.FFTLoss()], weights=[1, 1, 1])

    @pytest.mark.parametrize(
        ('losses', 'weights', 'dtype', 'shape'),
        [
            (COMBINATIONS[0], [1, 1], 'float64', (5, 128, 128, 3)),
            (COMBINATIONS[1], [1, 1], 'float64', (5, 128, 128, 3)),
        ],
    )
    def test_dtype_and_shape(
        self,
        losses: list,
        weights: list[float],
        dtype: str,
        shape: tuple[int, ...],
    ) -> None:
        y_true = tf.random.normal(shape=shape)
        y_pred = tf.random.normal(shape=shape)
        loss = cl.CombinedLoss(
            losses=losses,
            weights=weights,
            dtype=dtype,
        )(y_true, y_pred)
        assert loss.dtype.name == dtype
        assert loss.shape == (shape[0], )

    @pytest.mark.parametrize(
        ('losses', 'weights', 'shape'),
        [
            (COMBINATIONS[0], [1, 1], (5, 128, 128, 3)),
            (COMBINATIONS[1], [1, 1], (5, 128, 128, 3)),
        ],
    )
    def test_min_loss(
        self,
        losses: list,
        weights: list[float],
        shape: tuple[int, ...],
    ) -> None:
        y_true = tf.random.normal(shape=shape)
        loss = cl.CombinedLoss(
            losses=losses,
            weights=weights,
        )(y_true, y_true)
        assert np.allclose(loss.numpy(), np.zeros(shape[0]))
