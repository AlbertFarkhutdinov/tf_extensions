import numpy as np
import pytest
import tensorflow as tf

import tf_extensions.custom_losses as cl
from tf_extensions.custom_losses.base_loss import BaseLoss

loss_types = [
    cl.AdaptiveNMAE,
    cl.DISTS,
    cl.DSSIM,
    cl.FFTLoss,
    cl.MultiScaleDSSIM,
    cl.SoftDiceLoss,
    cl.VGGBase,
    cl.VGGLoss,
    cl.CustomLoss,
]
dtypes = [
    'float16',
    'float32',
    'float64',
]
tensor_shapes = [
    (5, 128, 128, 3),
    (5, 256, 256, 3),
]


default_config = {
    'reduction': 'none',
    'dtype': 'float32',
    'cls_name': 'BaseLoss',
    'is_normalized': False,
    'name': None,
}
non_default_config = {
    'dtype': 'float64',
    'is_normalized': True,
}

tensor2d1 = tf.constant(
    value=[
        [20, -20],
        [-20, 20],
    ],
)
tensor2d2 = tf.constant(
    value=[
        [-15, 15],
        [15, -15],
    ],
)

tensor2d1n = tf.constant(
    value=[
        [1, -1],
        [-1, 1],
    ],
)
tensor2d2n = tf.constant(
    value=[
        [-0.75, 0.75],
        [0.75, -0.75],
    ],
)

tensor4d1 = tf.constant(
    value=[
        [[[20], [0]], [[-20], [0]]],
        [[[0], [0]], [[0], [0]]],
    ],
)
tensor4d2 = tf.constant(
    value=[
        [[[-15], [0]], [[15], [0]]],
        [[[15], [0]], [[-15], [0]]],
    ],
)

tensor4d1n = tf.constant(
    value=[
        [[[2.0], [0]], [[-2.0], [0]]],
        [[[0], [0]], [[0], [0]]],
    ],
)
tensor4d2n = tf.constant(
    value=[
        [[[-1.5], [0]], [[1.5], [0]]],
        [[[15], [0]], [[-15], [0]]],
    ],
)


class TestBaseLoss:
    """Class for the Adaptive NMAE tests."""

    def test_default_get_config(self) -> None:
        loss = BaseLoss()
        assert loss.get_config() == default_config

    def test_default_from_config(self) -> None:
        loss = BaseLoss.from_config(non_default_config)
        for attr_name, attr_value in loss.get_config().items():
            if attr_name in non_default_config:
                assert attr_value == non_default_config[attr_name]
            else:
                assert attr_value == default_config[attr_name]

    @pytest.mark.parametrize(
        ('loss_type',),
        [
            (loss_type,)
            for loss_type in loss_types
        ],
    )
    def test_default_init(self, loss_type) -> None:
        loss = loss_type()
        assert loss.config.dtype == 'float32'

    @pytest.mark.parametrize(
        ('loss_type', 'name', 'dtype'),
        [
            (loss_type, 'random_name', dtype)
            for loss_type in loss_types
            for dtype in dtypes
            if not (dtype == 'float16' and 'FFT' in loss_type.__name__)
        ],
    )
    def test_init(
        self,
        loss_type,
        name: str,
        dtype: str,
    ) -> None:
        loss = loss_type(
            name=name,
            dtype=dtype,
        )
        assert loss.config.name == name
        assert loss.config.dtype == dtype

    @pytest.mark.parametrize(
        ('loss_type', 'dtype'),
        [
            (loss_type, dtype)
            for loss_type in loss_types
            for dtype in dtypes
            if not (dtype == 'float16' and 'FFT' in loss_type.__name__)
        ],
    )
    def test_dtype(
        self,
        loss_type,
        dtype: str,
    ) -> None:
        loss = loss_type(dtype=dtype)(
            tf.random.normal(shape=tensor_shapes[0]),
            tf.random.normal(shape=tensor_shapes[0]),
        )
        assert loss.dtype.name == dtype

    @pytest.mark.parametrize(
        ('loss_type', 'shape'),
        [
            (loss_type, shape)
            for loss_type in loss_types
            for shape in tensor_shapes
        ],
    )
    def test_shape(
        self,
        loss_type,
        shape: tuple[int, ...],
    ) -> None:
        loss = loss_type()(
            tf.random.normal(shape=shape),
            tf.random.normal(shape=shape),
        )
        assert loss.shape == (shape[0], )

    @pytest.mark.parametrize(
        ('loss_type', 'shape'),
        [
            (loss_type, shape)
            for loss_type in loss_types
            for shape in tensor_shapes
        ],
    )
    def test_min_loss(
        self,
        loss_type,
        shape: tuple[int, ...],
    ) -> None:
        y_true = tf.random.normal(shape=shape)
        loss = loss_type()(y_true, y_true)
        assert np.allclose(loss.numpy(), np.zeros(shape[0]))

    @pytest.mark.parametrize(
        ('loss_type', 'tensor1', 'tensor2', 'tensor1n', 'tensor2n'),
        [
            (loss_type, *tensors)
            for loss_type in loss_types
            for tensors in [
                # (tensor2d1, tensor2d2, tensor2d1n, tensor2d2n),
                (tensor4d1, tensor4d2, tensor4d1n, tensor4d2n),
            ]
        ],
    )
    def test_normalize_images(
        self,
        loss_type,
        tensor1,
        tensor2,
        tensor1n,
        tensor2n,
    ) -> None:
        y_true, y_pred = loss_type(
            is_normalized=True,
        ).normalize_images(
            y_true=tensor1,
            y_pred=tensor2,
        )
        assert np.allclose(y_true.numpy(), tensor1n)
        assert np.allclose(y_pred.numpy(), tensor2n)
