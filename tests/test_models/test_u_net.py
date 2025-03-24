from itertools import product

import numpy as np
import pytest
import tensorflow as tf

from tf_extensions import layers as cl
from tf_extensions.auxiliary.exceptions import EvenKernelError
from tf_extensions.layers import conv_configs as cc
from tf_extensions.models.u_net import UNet, UNetConfig

u_net_configs = [
    UNetConfig(
        without_reducing_filters=combination[0][0],
        is_partial_reducing=combination[0][1],
        first_kernel_size=combination[1],
        max_filters_number=combination[2],
        conv_block_config=cc.ConvolutionalBlockConfig(with_bn=combination[3]),
    )
    for combination in product(
        (
            (False, False),
            (True, False),
            (True, True),
        ),
        (None, (7, 7)),
        (None, 64),
        (True, False),
    )
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
def_u_net = {
    'conv_block_config': def_conv_block,
    'initial_filters_number': 16,
    'max_filters_number': None,
    'path_length': 4,
    'pooling': 2,
    'without_reducing_filters': False,
    'is_partial_reducing': True,
    'out_residual_blocks_number': 0,
    'is_skipped_with_concat': True,
    'first_kernel_size': None,
    'vector_length': None,
    'is_binary_classification': False,
    'with_attention': False,
    'with_variable_kernel': False,
    'first_blocks_without_dropout': 0,
    'name': 'u_net',
    'include_top': True,
}


class TestUNetConfig:

    def test_init(self) -> None:
        config = UNetConfig()
        assert config.conv_block_config == cc.ConvolutionalBlockConfig()
        filters_number = config.initial_filters_number
        wrf = config.without_reducing_filters
        out = config.out_residual_blocks_number
        skip = config.is_skipped_with_concat
        assert filters_number == def_u_net['initial_filters_number']
        assert config.path_length == def_u_net['path_length']
        assert config.pooling == def_u_net['pooling']
        assert wrf == def_u_net['without_reducing_filters']
        assert out == def_u_net['out_residual_blocks_number']
        assert skip == def_u_net['is_skipped_with_concat']
        assert config.first_kernel_size == def_u_net['first_kernel_size']
        assert config.vector_length == def_u_net['vector_length']

    def test_as_dict(self) -> None:
        config = UNetConfig()
        assert config.as_dict() == def_u_net

    def test_from_dict(self) -> None:
        config = UNetConfig()
        assert config.from_dict(properties=def_u_net) == config

    def test_config_name(self) -> None:
        unet_config = UNetConfig(
            with_attention=True,
            without_reducing_filters=True,
            is_partial_reducing=False,
            first_blocks_without_dropout=2,
            out_residual_blocks_number=1,
            first_kernel_size=(7, 7),
            vector_length=128,
        )
        config_name = unet_config.get_config_name()
        assert config_name == '_'.join(
            [
                'u_net',
                'input_neurons16',
                'relu2',
                'kernel3x3',
                'encoder4',
                'attention',
                'without_reducing_filters',
                '2without_drop',
                'out_res1',
                'concat',
                'first_kernel7x7',
                'vector_length128',
            ],
        )


class TestUNet:

    def test_init_without_args(self) -> None:
        model = UNet()
        assert isinstance(model.config, UNetConfig)

    @pytest.mark.parametrize(
        ('filters', 'first_kernel_size'),
        [
            (64, (2, 2)),
            (64, (4, 4)),
        ],
    )
    def test_init_fail(
        self,
        filters: int,
        first_kernel_size: tuple[int, ...],
    ) -> None:
        with pytest.raises(
            EvenKernelError,
            match='Even kernel is found:',
        ):
            UNet(
                initial_filters_number=filters,
                first_kernel_size=first_kernel_size,
            )

    @pytest.mark.parametrize('config', u_net_configs)
    def test_init(self, config: UNetConfig) -> None:
        model = UNet(**config.as_dict())
        assert model.config == config
        length = config.path_length
        pooling = config.pooling
        assert np.all(model.powers == np.arange(length))
        assert len(model.max_pools) == length
        assert len(model.encoder_layers) == length
        assert len(model.decoder_layers) == length
        assert len(model.conv_transpose_layers) == length
        for i in range(length):
            assert isinstance(model.max_pools[i], tf.keras.layers.MaxPooling2D)
            assert model.max_pools[i].pool_size == (pooling, pooling)
            assert model.max_pools[i].padding == 'same'
            assert isinstance(model.encoder_layers[i], cl.ConvolutionalBlock)
            assert isinstance(model.decoder_layers[i], cl.ConvolutionalBlock)
            assert isinstance(
                model.conv_transpose_layers[i],
                tf.keras.layers.Conv2DTranspose,
            )
        if config.conv_block_config.with_bn:
            assert len(model.decoder_bn_layers) == length
        else:
            assert not len(model.decoder_bn_layers)

        assert isinstance(model.middle_pair, cl.ConvolutionalBlock)
        assert isinstance(
            model.output_skipped_connections,
            cl.SkippedConnections,
        )

    @pytest.mark.parametrize(
        ('shape', 'config'),
        [((3, 128, 128, 1), config) for config in u_net_configs],
    )
    def test_call(
        self,
        shape: tuple[int, ...],
        config: UNetConfig,
    ) -> None:
        model = UNet(**config.as_dict())
        output = model.call(inputs=tf.random.normal(shape=shape))
        exp_shape = shape
        assert output.shape == exp_shape
