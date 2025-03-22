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

    def test_config_name(self) -> None:
        config = BaseNetConfig()
        config_name = config.get_config_name()
        assert config_name == 'base_net'


class TestBaseNet:

    def test_init_default(self) -> None:
        model = BaseNet()
        assert isinstance(model.config, BaseNetConfig)
