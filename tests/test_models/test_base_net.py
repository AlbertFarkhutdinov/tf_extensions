import pytest

from tf_extensions.models.base_net import BaseNet, BaseNetConfig

def_base_net = {
    'name': 'base_net',
    'include_top': True,
}


class TestBaseNetConfig:

    def test_init_default(self) -> None:
        config = BaseNetConfig()
        assert config.name == 'base_net'
        assert config.include_top

    def test_as_dict(self) -> None:
        config = BaseNetConfig()
        assert config.as_dict() == def_base_net

    def test_from_dict(self) -> None:
        config = BaseNetConfig()
        assert config.from_dict(properties=def_base_net) == config

    @pytest.mark.parametrize(
        ('name', 'include_top', 'expected'),
        [
            ('base_net', True, 'base_net'),
            ('base_net', False, 'base_net_without_top'),
        ],
    )
    def test_config_name(
        self,
        name: str,
        include_top: bool,
        expected: str,
    ) -> None:
        config = BaseNetConfig(
            name=name,
            include_top=include_top,
        )
        config_name = config.get_config_name()
        assert config_name == expected


class TestBaseNet:

    def test_init_default(self) -> None:
        model = BaseNet()
        assert isinstance(model.config, BaseNetConfig)
