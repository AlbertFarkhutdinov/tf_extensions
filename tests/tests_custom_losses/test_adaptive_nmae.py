import pytest

from tf_extensions.custom_losses.adaptive_nmae import AdaptiveNMAE

default_config = {
    'reduction': 'none',
    'dtype': 'float32',
    'is_normalized': False,
    'cls_name': 'AdaptiveNMAE',
    'name': 'adaptive_nmae',
    'scale': 13.5,
}
non_default_config = {
    'dtype': 'float64',
    'is_normalized': True,
    'cls_name': 'AdaptiveNMAE',
    'scale': 1,
}


class TestAdaptiveNMAE:
    """Class for the Adaptive NMAE tests."""

    def test_default_init(self) -> None:
        loss = AdaptiveNMAE()
        assert loss.config.scale == 13.5
        assert loss.config.name == 'adaptive_nmae'

    def test_default_get_config(self) -> None:
        loss = AdaptiveNMAE()
        assert loss.get_config() == default_config

    def test_default_from_config(self) -> None:
        loss = AdaptiveNMAE.from_config(non_default_config)
        for attr_name, attr_value in loss.get_config().items():
            if attr_name in non_default_config:
                assert attr_value == non_default_config[attr_name]
            else:
                assert attr_value == default_config[attr_name]

    @pytest.mark.parametrize(
        ('scale',),
        [
            (13.5,),
            (1,),
        ],
    )
    def test_init(
        self,
        scale: float,
    ) -> None:
        loss = AdaptiveNMAE(scale=scale)
        assert loss.config.scale == scale
