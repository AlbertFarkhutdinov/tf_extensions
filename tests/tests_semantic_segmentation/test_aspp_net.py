import numpy as np
import pytest
import tensorflow as tf

from tf_extensions.semantic_segmentation import configs as cfg
from tf_extensions.semantic_segmentation import custom_layers as cl
from tf_extensions.semantic_segmentation.aspp import ASPPNet

aspp_net_properties = [
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


class TestASPPNet:

    def test_init_without_args(self):
        model = ASPPNet()
        assert isinstance(model.config, cfg.CustomSegmentationNetConfig)

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
        aspp_net_properties,
    )
    def test_init(
        self,
        filters,
        kernel,
        activation,
        bias,
        bn,
        dropout,
        initializer,
    ):
        model = ASPPNet(
            config=cfg.CustomSegmentationNetConfig(
                initial_filters_number=filters,
                conv_block_config=cfg.ConvolutionalBlockConfig(
                    conv2d_config=cfg.Conv2DConfig(
                        kernel_size=kernel,
                        use_bias=bias,
                        kernel_initializer=initializer,
                    ),
                    activation=activation,
                    with_bn=bn,
                    with_dropout=dropout,
                ),
            ),
        )
        scales = [1, 2, 3, 4, 4, 3]
        for i, conv_block in enumerate((
            model.conv_pair1,
            model.conv_pair2,
            model.conv_pair3,
            model.conv_pair4,
            model.conv_pair5,
            model.conv_pair6,
        )):
            assert isinstance(conv_block, cl.ConvolutionalBlock)
            assert conv_block.filters == filters * scales[i]
            assert conv_block.config.with_bn == bn
            assert conv_block.config.with_dropout == dropout
            assert conv_block.config.activation == activation
            conv2d_config = conv_block.config.conv2d_config
            assert conv2d_config.padding == 'same'
            assert conv2d_config.use_bias == bias
            assert conv2d_config.kernel_initializer == initializer
        for conv_layer in (
            model.conv_middle,
            model.conv_out,
        ):
            assert isinstance(conv_layer, tf.keras.layers.Conv2D)
            assert conv_layer.padding == 'same'
            assert conv_layer.use_bias == bias
            assert conv_layer.kernel_size == (1, 1)
        assert model.conv_middle.filters == 48
        assert model.conv_out.filters == 1
        assert isinstance(model.aspp, cl.ASPPLayer)
        assert model.aspp.conv_kwargs['filters'] == 256
        assert model.aspp.conv_kwargs['padding'] == 'same'
        assert model.aspp.conv_kwargs['activation'] == 'relu'
        assert len(model.aspp.dilated_layers) == 4
        for layer_id, layer in enumerate(model.aspp.dilated_layers):
            if layer_id:
                assert layer.kernel_size == kernel
                assert layer.dilation_rate == (6 * layer_id, 6 * layer_id)
            else:
                assert layer.kernel_size == (1, 1)
            assert layer.filters == 256
            assert layer.padding == 'same'
        assert model.aspp.conv_out.filters == 256
        assert model.aspp.conv_out.kernel_size == (1, 1)
        assert model.aspp.conv_out.padding == 'same'

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
        [((3, 128, 128, 1), *item) for item in aspp_net_properties],
    )
    def test_call(
        self,
        input_shape,
        filters,
        kernel,
        activation,
        bias,
        bn,
        dropout,
        initializer,
    ):
        model = ASPPNet(
            config=cfg.CustomSegmentationNetConfig(
                initial_filters_number=filters,
                conv_block_config=cfg.ConvolutionalBlockConfig(
                    conv2d_config=cfg.Conv2DConfig(
                        kernel_size=kernel,
                        use_bias=bias,
                        kernel_initializer=initializer,
                    ),
                    activation=activation,
                    with_bn=bn,
                    with_dropout=dropout,
                ),
            ),
        )
        output = model.call(inputs=np.random.random(input_shape))
        assert output.shape == input_shape
