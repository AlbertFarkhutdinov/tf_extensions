from tf_extensions.custom_losses.soft_dice import SoftDiceLoss

default_config = {
    'reduction': 'none',
    'dtype': 'float32',
    'is_normalized': False,
    'cls_name': 'SoftDiceLoss',
    'name': 'sdl',
}
non_default_config = {
    'dtype': 'float64',
    'is_normalized': True,
    'cls_name': 'SoftDiceLoss',
}


class TestSoftDiceLoss:
    """Class for the SoftDiceLoss tests."""

    def test_default_init(self) -> None:
        loss = SoftDiceLoss()
        assert loss.name == 'sdl'
        assert loss.config.name == 'sdl'

    def test_default_get_config(self) -> None:
        loss = SoftDiceLoss()
        assert loss.get_config() == default_config

    def test_default_from_config(self) -> None:
        loss = SoftDiceLoss.from_config(non_default_config)
        for attr_name, attr_value in loss.get_config().items():
            if attr_name in non_default_config:
                assert attr_value == non_default_config[attr_name]
            else:
                assert attr_value == default_config[attr_name]
