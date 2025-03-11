import numpy as np
import pytest
import tensorflow as tf

from tf_extensions import layers as cl
from tf_extensions.layers import conv_configs as cc


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
        filters: int,
        layers_number: int,
        with_bn: bool,
        with_dropout: bool,
        activation: str,
    ) -> None:
        block = cl.ConvolutionalBlock(
            filters=filters,
            config=cc.ConvolutionalBlockConfig(
                conv2d_config=cc.Conv2DConfig(),
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
        inputs: tf.Tensor,
        exp_shape: tuple[int, ...],
        filters: int,
        layers_number: int,
        bn: bool,
        dropout: bool,
        activation: str,
    ) -> None:
        output = cl.ConvolutionalBlock(
            filters=filters,
            config=cc.ConvolutionalBlockConfig(
                conv2d_config=cc.Conv2DConfig(),
                layers_number=layers_number,
                activation=activation,
                with_bn=bn,
                with_dropout=dropout,
            ),
        )(inputs=inputs)
        assert output.shape == exp_shape
