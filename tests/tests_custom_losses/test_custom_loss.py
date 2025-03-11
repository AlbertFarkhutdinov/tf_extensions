import pytest

from tf_extensions.custom_losses.adaptive_nmae import AdaptiveNMAE
from tf_extensions.custom_losses.custom_loss import (
    CustomLoss,
    CustomLossConfig,
)
from tf_extensions.custom_losses.ms_dssim import MultiScaleDSSIM


class TestCustomLossConfig:
    """Class for the CustomLossConfig tests."""

    def test_post_init(self) -> None:
        with pytest.raises(
            ValueError,
            match=r'SSIM weight (.*) is out of range \[0; 1\].',
        ):
            CustomLossConfig(ssim_weight=-3)


class TestCustomLoss:
    """Class for the CustomLoss tests."""

    def test_default_init(self) -> None:
        loss = CustomLoss()
        default_nmae = AdaptiveNMAE()
        default_ms_sim = MultiScaleDSSIM()
        assert loss.config.name == 'custom_loss'
        assert loss.config.ssim_weight == 0.85
        assert type(loss.adaptive_nmae) is type(default_nmae)
        assert type(loss.ms_dssim) is type(default_ms_sim)
        assert loss.adaptive_nmae.get_config() == default_nmae.get_config()
        assert loss.ms_dssim.get_config() == default_ms_sim.get_config()

    def test_init_from_losses(self) -> None:
        default_nmae = AdaptiveNMAE()
        default_ms_sim = MultiScaleDSSIM()
        loss = CustomLoss(
            adaptive_nmae=default_nmae,
            ms_dssim=default_ms_sim,
        )
        assert loss.config.name == 'custom_loss'
        assert loss.config.ssim_weight == 0.85
        assert type(loss.adaptive_nmae) is type(default_nmae)
        assert type(loss.ms_dssim) is type(default_ms_sim)
        assert loss.adaptive_nmae.get_config() == default_nmae.get_config()
        assert loss.ms_dssim.get_config() == default_ms_sim.get_config()

    def test_init_from_configs(self) -> None:
        default_nmae = AdaptiveNMAE()
        default_ms_sim = MultiScaleDSSIM()
        loss = CustomLoss(
            adaptive_nmae=default_nmae.get_config(),
            ms_dssim=default_ms_sim.get_config(),
        )
        assert loss.config.name == 'custom_loss'
        assert loss.config.ssim_weight == 0.85
        assert type(loss.adaptive_nmae) is type(default_nmae)
        assert type(loss.ms_dssim) is type(default_ms_sim)
        assert loss.adaptive_nmae.get_config() == default_nmae.get_config()
        assert loss.ms_dssim.get_config() == default_ms_sim.get_config()

    def test_default_get_config(self) -> None:
        loss = CustomLoss()
        default_nmae = AdaptiveNMAE()
        default_ms_sim = MultiScaleDSSIM()
        assert loss.get_config() == {
            'reduction': 'none',
            'dtype': 'float32',
            'cls_name': 'CustomLoss',
            'is_normalized': False,
            'name': 'custom_loss',
            'ssim_weight': 0.85,
            'adaptive_nmae': default_nmae.get_config(),
            'ms_dssim': default_ms_sim.get_config(),
        }
