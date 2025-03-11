from itertools import product

import numpy as np
import pytest
import tensorflow as tf

from tf_extensions.semantic_segmentation.configs import (
    ConvolutionalBlockConfig,
    UNetConfig,
)
from tf_extensions.semantic_segmentation.custom_layers import (
    # UNetOutputLayer,
    ConvolutionalBlock,
    OutputSkippedConnections,
)
from tf_extensions.semantic_segmentation.u_net import UNet

u_net_configs = [
    UNetConfig(
        without_reducing_filters=combination[0][0],
        is_partial_reducing=combination[0][1],
        first_kernel_size=combination[1],
        max_filters_number=combination[2],
        conv_block_config=ConvolutionalBlockConfig(with_bn=combination[3]),
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


class TestUNet:

    def test_init_without_args(self):
        model = UNet()
        assert isinstance(model.config, UNetConfig)

    @pytest.mark.parametrize(
        ('filters', 'first_kernel_size'),
        [
            (64, (2, 2)),
            (64, (4, 4)),
        ],
    )
    def test_init_fail(self, filters, first_kernel_size):
        with pytest.raises(
            ValueError,
            match='Odd `first_kernel_size` is recommended.',
        ):
            UNet(
                config=UNetConfig(
                    initial_filters_number=filters,
                    first_kernel_size=first_kernel_size,
                ),
            )

    @pytest.mark.parametrize('config', u_net_configs)
    def test_init(self, config):
        model = UNet(config=config)
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
            assert isinstance(model.encoder_layers[i], ConvolutionalBlock)
            assert isinstance(model.decoder_layers[i], ConvolutionalBlock)
            assert isinstance(
                model.conv_transpose_layers[i],
                tf.keras.layers.Conv2DTranspose,
            )
        if config.conv_block_config.with_bn:
            assert len(model.decoder_bn_layers) == length
        else:
            assert not len(model.decoder_bn_layers)

        assert isinstance(model.middle_pair, ConvolutionalBlock)
        assert isinstance(
            model.output_skipped_connections,
            OutputSkippedConnections,
        )
        # assert isinstance(model.out_layer, UNetOutputLayer)

    @pytest.mark.parametrize(
        ('shape', 'config'),
        [((3, 128, 128, 1), config) for config in u_net_configs],
    )
    def test_call(self, shape, config):
        model = UNet(config=config)
        output = model.call(inputs=np.random.random(shape))
        exp_shape = shape
        assert output.shape == exp_shape
