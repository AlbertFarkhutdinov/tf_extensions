import numpy as np
import pytest
import tensorflow as tf

from tf_extensions.layers import ASPPLayer


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
