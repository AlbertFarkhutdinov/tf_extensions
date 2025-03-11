import numpy as np
import pytest
import tensorflow as tf

from tf_extensions import layers as cl
from tf_extensions.layers import conv_configs as cc


class TestSkippedConnections:

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
            (16, cc.ConvolutionalBlockConfig(), 3, True),
            (16, cc.ConvolutionalBlockConfig(), 3, False),
            (16, cc.ConvolutionalBlockConfig(), 0, True),
        ],
    )
    def test_init(
        self,
        filters: int,
        config: cc.ConvolutionalBlockConfig,
        is_skipped_concat: bool,
        blocks_number: int,
    ) -> None:
        layer = cl.SkippedConnections(
            filters=filters,
            config=config,
            is_skipped_with_concat=is_skipped_concat,
            blocks_number=blocks_number,
        )
        assert layer.filters == filters
        assert isinstance(layer.config, cc.ConvolutionalBlockConfig)
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
                cc.ConvolutionalBlockConfig(),
                2,
                True,
                (3, 128, 128, 48),
            ),
            (
                (3, 128, 128, 16),
                16,
                cc.ConvolutionalBlockConfig(),
                2,
                False,
                (3, 128, 128, 16),
            ),
            (
                (3, 128, 128, 16),
                16,
                cc.ConvolutionalBlockConfig(),
                0,
                True,
                (3, 128, 128, 16),
            ),
        ],
    )
    def test_call(
        self,
        input_shape: tuple[int, ...],
        filters: int,
        config: cc.ConvolutionalBlockConfig,
        is_skipped_concat: bool,
        blocks_number: int,
        exp_shape: tuple[int, ...],
    ) -> None:
        output = cl.SkippedConnections(
            filters=filters,
            config=config,
            is_skipped_with_concat=is_skipped_concat,
            blocks_number=blocks_number,
        )(inputs=np.random.random(input_shape))
        assert output.shape == exp_shape
