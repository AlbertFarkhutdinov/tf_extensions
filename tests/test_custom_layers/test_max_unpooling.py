import numpy as np
import tensorflow as tf
import pytest

from tf_extensions.layers import MaxUnpooling2D


class TestMaxUnpooling2D:

    def test_init(self) -> None:
        assert MaxUnpooling2D().pool_size == (2, 2)

    @pytest.mark.parametrize(
        ('input_shape', 'expected'),
        [
            ([(2, 64, 64, 1), (2, 64, 64, 1)], (2, 128, 128, 1)),
        ],
    )
    def test_compute_output_shape(
        self,
        input_shape: list[tuple[int, ...]],
        expected: tuple[int, ...],
    ) -> None:
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
    def test_call(
        self,
        mp_values: tf.Tensor,
        mp_indices: tf.Tensor,
        expected: tf.Tensor,
    ) -> None:
        unpooled = MaxUnpooling2D()(inputs=[mp_values, mp_indices])
        assert np.allclose(unpooled, expected)
