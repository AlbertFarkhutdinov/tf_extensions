import numpy as np
import pytest
import tensorflow as tf

from tf_extensions.semantic_segmentation.configs import (
    Conv2DConfig,
    ConvolutionalBlockConfig,
)
from tf_extensions.semantic_segmentation.custom_layers import (
    ASPPLayer,
    ConvolutionalBlock,
    MaxPoolingWithArgmax2D,
    MaxUnpooling2D,
    OutputSkippedConnections,
    UNetOutputLayer,
)


class TestMaxPoolingWithArgmax2D:

    def test_init(self):
        pooling_layer = MaxPoolingWithArgmax2D()
        assert pooling_layer.pool_size == (2, 2)
        assert pooling_layer.strides == (2, 2)
        assert pooling_layer.padding == 'same'

    def test_compute_mask(self):
        assert MaxPoolingWithArgmax2D().compute_mask(
            inputs=tf.random.normal(shape=(2, 128, 128, 1)),
        ) == [None, None]

    @pytest.mark.parametrize(
        ('input_shape', 'expected'),
        [
            ((2, 128, 128, 1), [(2, 64, 64, 1), (2, 64, 64, 1)]),
            ((2, 129, 129, 1), [(2, 64, 64, 1), (2, 64, 64, 1)]),
        ],
    )
    def test_compute_output_shape(self, input_shape, expected):
        assert MaxPoolingWithArgmax2D().compute_output_shape(
            input_shape=input_shape,
        ) == expected

    @pytest.mark.parametrize(
        ('inputs', 'expected_values', 'expected_indices'),
        [
            (
                np.array([
                    [
                        [[42, 40], [66, 11], [56, 68], [78, 78]],
                        [[28, 50], [90, 22], [6, 15], [37, 16]],
                        [[14, 83], [12, 54], [47, 68], [39, 55]],
                        [[12, 60], [19, 30], [10, 72], [28, 10]],
                    ],
                    [
                        [[54, 77], [88, 27], [27, 86], [34, 34]],
                        [[76, 72], [86, 7], [86, 68], [39, 19]],
                        [[97, 59], [42, 59], [46, 87], [43, 20]],
                        [[74, 86], [96, 28], [34, 37], [32, 24]],
                    ],
                ]),
                np.array([
                    [[[90, 50], [78, 78]], [[19, 83], [47, 72]]],
                    [[[88, 77], [86, 86]], [[97, 86], [46, 87]]],
                ]),
                np.array([
                    [[[10, 9], [6, 7]], [[26, 17], [20, 29]]],
                    [[[2, 1], [12, 5]], [[16, 25], [20, 21]]],
                ]),
            ),
        ],
    )
    def test_call(self, inputs, expected_values, expected_indices):
        mp_values, mp_indices = MaxPoolingWithArgmax2D()(inputs=inputs)
        assert np.allclose(mp_values, expected_values)
        assert np.allclose(mp_indices, expected_indices)


class TestMaxUnpooling2D:

    def test_init(self):
        assert MaxUnpooling2D().pool_size == (2, 2)

    @pytest.mark.parametrize(
        ('input_shape', 'expected'),
        [
            ([(2, 64, 64, 1), (2, 64, 64, 1)], (2, 128, 128, 1)),
        ],
    )
    def test_compute_output_shape(self, input_shape, expected):
        assert MaxUnpooling2D().compute_output_shape(
            input_shape=input_shape,
        ) == expected

    @pytest.mark.parametrize(
        ('mp_values', 'mp_indices', 'expected'),
        [
            (
                np.array([
                    [[[90, 50], [78, 78]], [[19, 83], [47, 72]]],
                    [[[88, 77], [86, 86]], [[97, 86], [46, 87]]],
                ]),
                np.array([
                    [[[10, 9], [6, 7]], [[26, 17], [20, 29]]],
                    [[[2, 1], [12, 5]], [[16, 25], [20, 21]]],
                ]),
                np.array([
                    [
                        [[0, 0], [0, 0], [0, 0], [78, 78]],
                        [[0, 50], [90, 0], [0, 0], [0, 0]],
                        [[0, 83], [0, 0], [47, 0], [0, 0]],
                        [[0, 0], [19, 0], [0, 72], [0, 0]],
                    ],
                    [
                        [[0, 77], [88, 0], [0, 86], [0, 0]],
                        [[0, 0], [0, 0], [86, 0], [0, 0]],
                        [[97, 0], [0, 0], [46, 87], [0, 0]],
                        [[0, 86], [0, 0], [0, 0], [0, 0]],
                    ],
                ]),
            ),
        ],
    )
    def test_call(self, mp_values, mp_indices, expected):
        unpooled = MaxUnpooling2D()(inputs=[mp_values, mp_indices])
        assert np.allclose(unpooled, expected)


class TestConvolutionalBlock:

    @pytest.mark.parametrize(
        ('filters', 'layers_number', 'with_bn', 'with_dropout', 'activation'),
        [
            (2, 2, False, True, 'relu'),
            (2, 2, False, False, 'relu'),
            (2, 2, True, True, 'relu'),
            (2, 2, True, False, 'relu'),
        ],
    )
    def test_init(
        self,
        filters,
        layers_number,
        with_bn,
        with_dropout,
        activation,
    ):
        block = ConvolutionalBlock(
            filters=filters,
            config=ConvolutionalBlockConfig(
                conv2d_config=Conv2DConfig(),
                layers_number=layers_number,
                activation=activation,
                with_bn=with_bn,
                with_dropout=with_dropout,
            ),
        )
        assert block.config.with_bn == with_bn
        assert block.config.with_dropout == with_dropout
        assert len(block.conv_layers) == block.config.layers_number
        assert block.config.layers_number == layers_number
        assert len(block.activations) == block.config.layers_number
        assert block.config.layers_number == layers_number
        if with_bn:
            assert len(block.normalizations) == layers_number
        else:
            assert not block.normalizations
        if with_dropout:
            assert len(block.dropouts) == layers_number - 1
        else:
            assert not block.dropouts
        for layer_id in range(layers_number):
            assert isinstance(
                block.conv_layers[layer_id],
                tf.keras.layers.Conv2D,
            )
            assert block.conv_layers[layer_id].filters == filters
            assert block.conv_layers[layer_id].kernel_size == (3, 3)
            assert block.conv_layers[layer_id].padding == 'same'

    @pytest.mark.parametrize(
        (
            'inputs',
            'exp_shape',
            'filters',
            'layers_number',
            'bn',
            'dropout',
            'activation',
        ),
        [
            (
                np.random.random((3, 4, 4, 1)),
                (3, 4, 4, 2),
                2,
                2,
                False,
                False,
                'relu',
            ),
            (
                np.random.random((3, 4, 4, 1)),
                (3, 4, 4, 2),
                2,
                2,
                False,
                True,
                'relu',
            ),
            (
                np.random.random((3, 4, 4, 1)),
                (3, 4, 4, 2),
                2,
                2,
                True,
                False,
                'relu',
            ),
            (
                np.random.random((3, 4, 4, 1)),
                (3, 4, 4, 2),
                2,
                2,
                True,
                True,
                'relu',
            ),
        ],
    )
    def test_call(
        self,
        inputs,
        exp_shape,
        filters,
        layers_number,
        bn,
        dropout,
        activation,
    ):
        output = ConvolutionalBlock(
            filters=filters,
            config=ConvolutionalBlockConfig(
                conv2d_config=Conv2DConfig(),
                layers_number=layers_number,
                activation=activation,
                with_bn=bn,
                with_dropout=dropout,
            ),
        )(inputs=inputs)
        assert output.shape == exp_shape


class TestASPPLayer:

    def test_init(self):
        filters_number = 2
        dilation_scale = 2
        layer = ASPPLayer(
            filters_number=filters_number,
            dilation_scale=dilation_scale,
        )
        assert isinstance(layer.dilated_layers, list)
        assert isinstance(layer.dilated_layers[0], tf.keras.layers.Conv2D)
        assert isinstance(layer.conv_out, tf.keras.layers.Conv2D)
        assert len(layer.dilated_layers) == 4
        assert layer.dilated_layers[0].kernel_size == (1, 1)
        assert layer.dilated_layers[1].kernel_size == (3, 3)
        assert layer.conv_out.kernel_size == (1, 1)
        assert layer.dilated_layers[0].filters == filters_number
        assert layer.dilated_layers[1].filters == filters_number
        assert layer.conv_out.filters == filters_number
        assert layer.dilated_layers[0].padding == 'same'
        assert layer.dilated_layers[1].padding == 'same'
        assert layer.conv_out.padding == 'same'
        assert layer.dilated_layers[1].dilation_rate == (
            dilation_scale,
            dilation_scale,
        )
        assert layer.dilated_layers[2].dilation_rate == (
            dilation_scale * 2,
            dilation_scale * 2,
        )

    @pytest.mark.parametrize(
        (
            'inputs',
            'filters',
            'kernel_size',
            'dil_scale',
            'dil_number',
            'expected_shape',
        ),
        [
            (
                np.random.random((3, 4, 4, 1)),
                2,
                (3, 3),
                2,
                3,
                (3, 4, 4, 2),
            ),
        ],
    )
    def test_call(
        self,
        inputs,
        filters,
        kernel_size,
        dil_scale,
        dil_number,
        expected_shape,
    ):
        output = ASPPLayer(
            filters_number=filters,
            kernel_size=kernel_size,
            dilation_scale=dil_scale,
            dilation_number=dil_number,
        )(inputs=inputs)
        assert output.shape == expected_shape


class TestOutputSkippedConnections:

    @pytest.mark.parametrize(
        (
            'filters',
            'config',
            'blocks_number',
            'is_skipped_concat',
        ),
        [
            (16, None, 3, True),
            (16, None, 3, False),
            (16, None, 0, True),
            (16, ConvolutionalBlockConfig(), 3, True),
            (16, ConvolutionalBlockConfig(), 3, False),
            (16, ConvolutionalBlockConfig(), 0, True),
        ],
    )
    def test_init(self, filters, config, is_skipped_concat, blocks_number):
        layer = OutputSkippedConnections(
            filters=filters,
            config=config,
            is_skipped_with_concat=is_skipped_concat,
            blocks_number=blocks_number,
        )
        assert layer.filters == filters
        assert isinstance(layer.config, ConvolutionalBlockConfig)
        assert layer.is_skipped_with_concat == is_skipped_concat
        assert layer.blocks_number == blocks_number
        assert len(layer.conv_layers) == layer.blocks_number
        for conv_layer in layer.conv_layers:
            assert isinstance(conv_layer, tf.keras.layers.Conv2D)

    @pytest.mark.parametrize(
        (
            'input_shape',
            'filters',
            'config',
            'blocks_number',
            'is_skipped_concat',
            'exp_shape',
        ),
        [
            ((3, 128, 128, 16), 16, None, 2, True, (3, 128, 128, 48)),
            ((3, 128, 128, 16), 16, None, 2, False, (3, 128, 128, 16)),
            ((3, 128, 128, 16), 16, None, 0, True, (3, 128, 128, 16)),
            (
                (3, 128, 128, 16),
                16,
                ConvolutionalBlockConfig(),
                2,
                True,
                (3, 128, 128, 48),
            ),
            (
                (3, 128, 128, 16),
                16,
                ConvolutionalBlockConfig(),
                2,
                False,
                (3, 128, 128, 16),
            ),
            (
                (3, 128, 128, 16),
                16,
                ConvolutionalBlockConfig(),
                0,
                True,
                (3, 128, 128, 16),
            ),
        ],
    )
    def test_call(
        self,
        input_shape,
        filters,
        config,
        is_skipped_concat,
        blocks_number,
        exp_shape,
    ):
        output = OutputSkippedConnections(
            filters=filters,
            config=config,
            is_skipped_with_concat=is_skipped_concat,
            blocks_number=blocks_number,
        )(inputs=np.random.random(input_shape))
        assert output.shape == exp_shape


class TestUNetOutputLayer:

    @pytest.mark.parametrize(
        (
            'vector_length',
            'conv2d_config',
        ),
        [
            (None, None),
            (None, Conv2DConfig()),
            (2, None),
            (2, Conv2DConfig()),
        ],
    )
    def test_init(self, vector_length, conv2d_config):
        layer = UNetOutputLayer(
            vector_length=vector_length,
            conv2d_config=conv2d_config,
        )
        assert layer.vector_length == vector_length
        assert isinstance(layer.conv2d_config, Conv2DConfig)
        if layer.vector_length:
            assert isinstance(layer.out_layer, tf.keras.layers.Conv1D)
        else:
            assert isinstance(layer.out_layer, tf.keras.layers.Conv2D)

    @pytest.mark.parametrize(
        (
            'input_shape',
            'vector_length',
            'conv2d_config',
            'exp_shape',
        ),
        [
            ((3, 128, 128, 1), None, None, (3, 128, 128, 1)),
            ((3, 128, 128, 1), None, Conv2DConfig(), (3, 128, 128, 1)),
            ((3, 128, 128, 1), 2, None, (3, 128, 1, 1)),
            ((3, 128, 128, 1), 2, Conv2DConfig(), (3, 128, 1, 1)),
        ],
    )
    def test_call(self, input_shape, vector_length, conv2d_config, exp_shape):
        output = UNetOutputLayer(
            vector_length=vector_length,
            conv2d_config=conv2d_config,
        )(inputs=np.random.random(input_shape))
        assert output.shape == exp_shape
