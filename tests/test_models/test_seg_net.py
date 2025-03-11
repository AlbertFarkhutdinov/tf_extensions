import numpy as np
import pytest
import tensorflow as tf

from tf_extensions.layers import conv_configs as cc
from tf_extensions.models.seg_net import SegNet, SegNetConfig

seg_net_properties = [
    (
        64,
        (3, 3),
        'relu',
        True,
        False,
        False,
        'glorot_uniform',
        4,
        2,
        2,
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
def_seg_net = {
    'conv_block_config': def_conv_block,
    'initial_filters_number': 16,
    'max_filters_number': None,
    'path_length': 4,
    'pooling': 2,
}


class TestSegNetConfig:

    def test_init(self):
        config = SegNetConfig()
        assert config.conv_block_config == cc.ConvolutionalBlockConfig()
        filters_number = config.initial_filters_number
        assert filters_number == def_seg_net['initial_filters_number']
        assert config.path_length == def_seg_net['path_length']
        assert config.pooling == def_seg_net['pooling']

    def test_as_dict(self):
        config = SegNetConfig()
        assert config.as_dict() == def_seg_net

    def test_from_dict(self):
        config = SegNetConfig()
        assert config.from_dict(properties=def_seg_net) == config

    def test_config_name(self):
        seg_net_config = SegNetConfig(
            path_length=5,
            pooling=3,
        )
        config_name = seg_net_config.get_config_name()
        assert config_name == (
            'encoder5_pooling3_input_neurons16_relu2_kernel3x3'
        )


class TestSegNet:

    def test_init_without_args(self):
        model = SegNet()
        assert isinstance(model.config, SegNetConfig)

    @pytest.mark.parametrize(
        (
            'filters',
            'kernel',
            'act',
            'bias',
            'bn',
            'dropout',
            'init',
            'length',
            'pooling',
            'layers',
        ),
        seg_net_properties,
    )
    def test_init(
        self,
        filters,
        kernel,
        act,
        bias,
        bn,
        dropout,
        init,
        length,
        pooling,
        layers,
    ):
        model = SegNet(
            config=SegNetConfig(
                initial_filters_number=filters,
                conv_block_config=cc.ConvolutionalBlockConfig(
                    conv2d_config=cc.Conv2DConfig(
                        kernel_size=kernel,
                        use_bias=bias,
                        kernel_initializer=init,
                    ),
                    layers_number=layers,
                    activation=act,
                    with_bn=bn,
                    with_dropout=dropout,
                ),
                path_length=length,
                pooling=pooling,
            ),
        )
        assert model.config.path_length == length
        assert model.config.pooling == pooling
        assert model.config.initial_filters_number == filters
        conv_block_config = model.config.conv_block_config
        assert conv_block_config.layers_number == layers
        assert conv_block_config.activation == act
        assert conv_block_config.with_bn is True
        assert conv_block_config.with_dropout == dropout
        assert conv_block_config.conv2d_config.kernel_size == kernel
        assert conv_block_config.conv2d_config.padding == 'same'
        assert conv_block_config.conv2d_config.use_bias == bias
        assert conv_block_config.conv2d_config.kernel_initializer == init

        assert np.all(model.powers == np.arange(length))
        assert isinstance(model.output_convolution, tf.keras.layers.Conv2D)
        assert model.output_convolution.filters == 2
        assert model.output_convolution.kernel_size == (1, 1)
        assert model.output_convolution.padding == 'same'
        assert isinstance(
            model.output_batch_normalization,
            tf.keras.layers.BatchNormalization,
        )
        assert len(model.max_pools) == length
        assert len(model.max_unpools) == length
        assert len(model.encoder_layers) == length
        assert len(model.decoder_layers) == length - 1

    @pytest.mark.parametrize(
        (
            'shape',
            'filters',
            'kernel',
            'act',
            'bias',
            'bn',
            'dropout',
            'init',
            'length',
            'pooling',
            'layers',
        ),
        [((3, 128, 128, 2), *item) for item in seg_net_properties],
    )
    def test_call(
        self,
        shape,
        filters,
        kernel,
        act,
        bias,
        bn,
        dropout,
        init,
        length,
        pooling,
        layers,
    ):
        model = SegNet(
            config=SegNetConfig(
                initial_filters_number=filters,
                conv_block_config=cc.ConvolutionalBlockConfig(
                    conv2d_config=cc.Conv2DConfig(
                        kernel_size=kernel,
                        use_bias=bias,
                        kernel_initializer=init,
                    ),
                    layers_number=layers,
                    activation=act,
                    with_bn=bn,
                    with_dropout=dropout,
                ),
                path_length=length,
                pooling=pooling,
            ),
        )
        output = model.call(inputs=tf.random.normal(shape=shape))
        assert output.shape == shape
