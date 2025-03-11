import numpy as np
import pytest
import tensorflow as tf

from tf_extensions.custom_losses.dists import DISTS, DISTSConfig

valid_layers = [
    ['block5_conv4'],
    [
        'block1_conv2',
        'block2_conv2',
        'block3_conv3',
        'block4_conv3',
        'block5_conv3',
    ],
]
texture_weights = [0.5, 0.3, 0.7]
dtypes = [
    'float16',
    'float32',
    'float64',
]
tensor_shapes = [
    (5, 64, 128, 3),
]
batch_sizes = [None, 64]


default_config = {
    'reduction': 'none',
    'dtype': 'float32',
    'cls_name': 'DISTS',
    'is_normalized': False,
    'name': 'dists',
    'layer_names': ['block5_conv4'],
    'texture_weight': 0.5,
    'texture_stability_constant': 1e-6,
    'structure_stability_constant': 1e-6,
    'batch_size': None,
}
non_default_config = {
    'dtype': 'float64',
    'cls_name': 'DISTS',
    'is_normalized': True,
    'layer_names': [
        'block1_conv2',
        'block2_conv2',
        'block3_conv3',
        'block4_conv3',
        'block5_conv3',
    ],
    'texture_weight': 0.3,
    'batch_size': 64,
}


class TestDISTSConfig:
    """Class for the DISTSConfig tests."""

    def test_post_init(self) -> None:
        with pytest.raises(
            ValueError,
            match=r'Texture weight (.*) is out of range \[0; 1\].',
        ):
            DISTSConfig(texture_weight=1.5)


class TestDISTS:
    """Class for the DISTS tests."""

    def test_default_init(self) -> None:
        dists = DISTS()
        assert dists.config.name == 'dists'
        assert dists.config.layer_names == ['block5_conv4']
        assert dists.config.texture_weight == 0.5
        assert dists.config.texture_stability_constant == 1e-6
        assert dists.config.structure_stability_constant == 1e-6

    def test_default_get_config(self) -> None:
        assert DISTS().get_config() == default_config

    def test_default_from_config(self) -> None:
        dists = DISTS.from_config(non_default_config)
        for attr_name, attr_value in dists.get_config().items():
            if attr_name in non_default_config:
                assert attr_value == non_default_config[attr_name]
            else:
                assert attr_value == default_config[attr_name]

    @pytest.mark.parametrize(
        ('layer_names', 'texture_weight'),
        [
            (layer_names, texture_weight)
            for layer_names in valid_layers
            for texture_weight in texture_weights
        ],
    )
    def test_init(
        self,
        layer_names: list[str],
        texture_weight: float,
    ) -> None:
        dists = DISTS(
            layer_names=layer_names,
            texture_weight=texture_weight,
        )
        assert dists.config.layer_names == layer_names
        assert dists.config.texture_weight == texture_weight

    @pytest.mark.parametrize(
        ('layer_names', 'texture_weight', 'dtype', 'batch_size'),
        [
            (layer_names, texture_weight, dtype, batch_size)
            for layer_names in valid_layers
            for texture_weight in texture_weights
            for dtype in dtypes
            for batch_size in batch_sizes
        ],
    )
    def test_dtype(
        self,
        layer_names: list[str],
        texture_weight: float,
        dtype: str,
        batch_size: int,
    ) -> None:
        dists = DISTS(
            layer_names=layer_names,
            texture_weight=texture_weight,
            dtype=dtype,
            batch_size=batch_size,
        )
        loss = dists(
            tf.random.normal(shape=tensor_shapes[0]),
            tf.random.normal(shape=tensor_shapes[0]),
        )
        assert loss.dtype.name == dtype

    @pytest.mark.parametrize(
        ('layer_names', 'texture_weight', 'shape', 'batch_size'),
        [
            (layer_names, texture_weight, shape, batch_size)
            for layer_names in valid_layers
            for texture_weight in texture_weights
            for shape in tensor_shapes
            for batch_size in batch_sizes
        ],
    )
    def test_shape(
        self,
        layer_names: list[str],
        texture_weight: float,
        shape: tuple[int, ...],
        batch_size: int,
    ) -> None:
        dists = DISTS(
            layer_names=layer_names,
            texture_weight=texture_weight,
            batch_size=batch_size,
        )
        loss = dists(
            tf.random.normal(shape=shape),
            tf.random.normal(shape=shape),
        )
        assert loss.shape == (shape[0], )

    @pytest.mark.parametrize(
        ('layer_names', 'texture_weight', 'shape', 'batch_size'),
        [
            (layer_names, texture_weight, shape, batch_size)
            for layer_names in valid_layers
            for texture_weight in texture_weights
            for shape in tensor_shapes
            for batch_size in batch_sizes
        ],
    )
    def test_min_loss(
        self,
        layer_names: list[str],
        texture_weight: float,
        shape: tuple[int, ...],
        batch_size: int,
    ) -> None:
        dists = DISTS(
            layer_names=layer_names,
            texture_weight=texture_weight,
            batch_size=batch_size,
        )
        y_true = tf.random.normal(shape=shape)
        loss = dists(y_true, y_true)
        assert np.allclose(loss.numpy(), np.zeros(shape[0]))
