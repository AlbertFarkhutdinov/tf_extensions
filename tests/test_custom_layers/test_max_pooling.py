import numpy as np
import pytest
import tensorflow as tf

from tf_extensions.layers import MaxPoolingWithArgmax2D


class TestMaxPoolingWithArgmax2D:

    def test_init(self) -> None:
        pooling_layer = MaxPoolingWithArgmax2D()
        assert pooling_layer.pool_size == (2, 2)
        assert pooling_layer.strides == (2, 2)
        assert pooling_layer.padding == 'same'

    def test_compute_mask(self) -> None:
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
    def test_compute_output_shape(
        self,
        input_shape: tuple[int, ...],
        expected: tuple[int, ...],
    ) -> None:
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
    def test_call(
        self,
        inputs: tf.Tensor,
        expected_values: tf.Tensor,
        expected_indices: tf.Tensor,
    ) -> None:
        mp_values, mp_indices = MaxPoolingWithArgmax2D()(inputs=inputs)
        assert np.allclose(mp_values, expected_values)
        assert np.allclose(mp_indices, expected_indices)
