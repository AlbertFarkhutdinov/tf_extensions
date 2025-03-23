import pytest

from tf_extensions.layers import ConvolutionalBlock
from tf_extensions.layers import conv_configs as cc
from tf_extensions.models.base_cnn import BaseCNN, BaseCNNConfig

base_net_properties = [
    (
        64,
        (3, 3),
        'relu',
        True,
        False,
        False,
        'glorot_uniform',
    ),
]
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
def_base_net = {
    'conv_block_config': def_conv_block,
    'initial_filters_number': 16,
    'max_filters_number': None,
    'include_top': True,
    'name': "base_cnn",
}


class TestBaseNetConfig:

    def test_init(self) -> None:
        config = BaseCNNConfig()
        assert config.conv_block_config == cc.ConvolutionalBlockConfig()
        filters_number = config.initial_filters_number
        assert filters_number == def_base_net['initial_filters_number']

    def test_as_dict(self) -> None:
        config = BaseCNNConfig()
        assert config.as_dict() == def_base_net

    def test_from_dict(self) -> None:
        config = BaseCNNConfig()
        assert config.from_dict(properties=def_base_net) == config

    def test_config_name(self) -> None:
        conv_block_config = cc.ConvolutionalBlockConfig()
        custom_seg_net_config = BaseCNNConfig(
            conv_block_config=conv_block_config,
            initial_filters_number=32,
            max_filters_number=64,
        )
        config_name = custom_seg_net_config.get_config_name()
        exp = 'base_cnn_input_neurons32_max_neurons64_relu2_kernel3x3'
        assert config_name == exp


class TestBaseNet:

    def test_init_without_args(self) -> None:
        model = BaseCNN()
        assert isinstance(model.config, BaseCNNConfig)

    @pytest.mark.parametrize(
        (
            'filters',
            'kernel',
            'activation',
            'bias',
            'bn',
            'dropout',
            'initializer',
        ),
        base_net_properties,
    )
    def test_init(
        self,
        filters: int,
        kernel: tuple[int, ...],
        activation: str,
        bias: bool,
        bn: bool,
        dropout: bool,
        initializer: str,
    ) -> None:
        model = BaseCNN(
            initial_filters_number=filters,
            conv_block_config=cc.ConvolutionalBlockConfig(
                conv2d_config=cc.Conv2DConfig(
                    kernel_size=kernel,
                    use_bias=bias,
                    kernel_initializer=initializer,
                ),
                activation=activation,
                with_bn=bn,
                with_dropout=dropout,
            ),
        )
        assert model.config.initial_filters_number == filters
        assert model.config.conv_block_config.layers_number == 2
        assert model.config.conv_block_config.activation == activation
        assert model.config.conv_block_config.with_bn == bn
        assert model.config.conv_block_config.with_dropout == dropout
        conv2d_config = model.config.conv_block_config.conv2d_config
        assert conv2d_config.kernel_size == kernel
        assert conv2d_config.padding == 'same'
        assert conv2d_config.use_bias == bias
        assert conv2d_config.kernel_initializer == initializer

    @pytest.mark.parametrize(
        ('filters', 'kernel_size'),
        [
            (64, (2, 2)),
            (64, (4, 4)),
        ],
    )
    def test_init_fail(
        self,
        filters: int,
        kernel_size: tuple[int, ...],
    ) -> None:
        with pytest.raises(
            ValueError,
            match='Odd `kernel_size` is recommended.',
        ):
            BaseCNN(
                initial_filters_number=filters,
                conv_block_config=cc.ConvolutionalBlockConfig(
                    conv2d_config=cc.Conv2DConfig(
                        kernel_size=kernel_size,
                    ),
                ),
            )

    @pytest.mark.parametrize(
        (
            'input_shape',
            'filters',
            'kernel',
            'activation',
            'bias',
            'bn',
            'dropout',
            'initializer',
        ),
        [((128, 128, 1), *item) for item in base_net_properties],
    )
    def test_build_graph(
        self,
        input_shape: tuple[int, ...],
        filters: int,
        kernel: tuple[int, ...],
        activation: str,
        bias: bool,
        bn: bool,
        dropout: bool,
        initializer: str,
    ) -> None:
        model = BaseCNN(
            initial_filters_number=filters,
            conv_block_config=cc.ConvolutionalBlockConfig(
                conv2d_config=cc.Conv2DConfig(
                    kernel_size=kernel,
                    use_bias=bias,
                    kernel_initializer=initializer,
                ),
                activation=activation,
                with_bn=bn,
                with_dropout=dropout,
            ),
        )
        graph = model.build_graph(input_shape=input_shape)
        assert graph.inputs[0].shape == (None, *input_shape)
        assert graph.outputs[0].shape == (None, *input_shape)

    @pytest.mark.parametrize(
        (
            'filter_scale',
            'conv_kernel',
            'filters',
            'kernel',
            'act',
            'bias',
            'bn',
            'dropout',
            'init',
            'is_dropout_off',
        ),
        [
            *[(2, None, *item, True) for item in base_net_properties],
            *[(2, (5, 5), *item, False) for item in base_net_properties],
        ],
    )
    def test_get_convolutional_pair(
        self,
        filter_scale: int,
        conv_kernel: tuple[int, ...],
        filters: int,
        kernel: tuple[int, ...],
        act: str,
        bias: bool,
        bn: bool,
        dropout: bool,
        init: str,
        is_dropout_off: bool,
    ) -> None:
        model = BaseCNN(
            initial_filters_number=filters,
            conv_block_config=cc.ConvolutionalBlockConfig(
                conv2d_config=cc.Conv2DConfig(
                    kernel_size=kernel,
                    use_bias=bias,
                    kernel_initializer=init,
                ),
                activation=act,
                with_bn=bn,
                with_dropout=dropout,
            ),
        )
        conv_block = model.get_convolutional_block(
            filter_scale=filter_scale,
            kernel_size=conv_kernel,
            is_dropout_off=is_dropout_off,
        )
        assert isinstance(conv_block, ConvolutionalBlock)
        assert conv_block.filters == filters * filter_scale
        assert conv_block.config.layers_number == 2
        assert conv_block.config.activation == act
        assert conv_block.config.with_bn == bn
        if is_dropout_off:
            assert not conv_block.config.with_dropout
        else:
            assert conv_block.config.with_dropout == dropout
        conv2d_config = conv_block.config.conv2d_config
        assert conv2d_config.kernel_size == conv_kernel or kernel
        assert conv2d_config.padding == 'same'
        assert conv2d_config.use_bias == bias
        assert conv2d_config.kernel_initializer == init
