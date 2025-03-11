import numpy as np
import pytest
import tensorflow as tf

from tf_extensions.semantic_segmentation.configs import (
    Conv2DConfig,
    ConvolutionalBlockConfig,
    SegNetConfig,
)
from tf_extensions.semantic_segmentation.seg_net import SegNet

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
                conv_block_config=ConvolutionalBlockConfig(
                    conv2d_config=Conv2DConfig(
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
                conv_block_config=ConvolutionalBlockConfig(
                    conv2d_config=Conv2DConfig(
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
        output = model.call(inputs=np.random.random(shape))
        assert output.shape == shape
