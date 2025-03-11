from tf_extensions.layers import conv_configs as cc

def_conv2d = {
    'kernel_size': (3, 3),
    'padding': 'same',
    'use_bias': True,
    'kernel_initializer': 'glorot_uniform',
}
def_conv_block = {
    'conv2d_config': def_conv2d,
    'drop_rate': 0.5,
    'layers_number': 2,
    'activation': 'relu',
    'with_bn': False,
    'with_dropout': False,
    'with_skipped': False,
}


class TestConv2DConfig:

    def test_init(self) -> None:
        config = cc.Conv2DConfig()
        assert config.kernel_size == def_conv2d['kernel_size']
        assert config.padding == def_conv2d['padding']
        assert config.use_bias == def_conv2d['use_bias']
        assert config.kernel_initializer == def_conv2d['kernel_initializer']

    def test_as_dict(self) -> None:
        config = cc.Conv2DConfig()
        assert config.as_dict() == def_conv2d

    def test_from_dict(self) -> None:
        config = cc.Conv2DConfig()
        assert config.from_dict(properties=def_conv2d) == config

    def test_config_name(self) -> None:
        config = cc.Conv2DConfig(
            kernel_size=(5, 5),
            padding='valid',
            use_bias=False,
            kernel_initializer='he_normal',
        )
        config_name = config.get_config_name()
        assert config_name == 'kernel5x5_pad_valid_without_bias_init_he_normal'


class TestConvolutionalBlockConfig:

    def test_init(self) -> None:
        config = cc.ConvolutionalBlockConfig()
        assert config.conv2d_config == cc.Conv2DConfig()
        assert config.layers_number == def_conv_block['layers_number']
        assert config.activation == def_conv_block['activation']
        assert config.with_bn == def_conv_block['with_bn']
        assert config.with_dropout == def_conv_block['with_dropout']

    def test_as_dict(self) -> None:
        config = cc.ConvolutionalBlockConfig()
        assert config.as_dict() == def_conv_block

    def test_from_dict(self) -> None:
        config = cc.ConvolutionalBlockConfig()
        assert config.from_dict(properties=def_conv_block) == config

    def test_config_name(self) -> None:
        conv2d_config = cc.Conv2DConfig(kernel_size=(5, 5))
        config = cc.ConvolutionalBlockConfig(
            conv2d_config=conv2d_config,
            layers_number=3,
            with_skipped=True,
            with_bn=True,
            with_dropout=True,
            drop_rate=0.3,
        )
        config_name = config.get_config_name()
        assert config_name == 'relu3_residual_bn_drop30_kernel5x5'
