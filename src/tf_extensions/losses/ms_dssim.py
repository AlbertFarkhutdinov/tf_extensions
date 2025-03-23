"""Module providing the Multi-Scale DSSIM loss function."""
from dataclasses import dataclass

import tensorflow as tf

from tf_extensions.losses.base_loss import BaseLoss
from tf_extensions.losses.dssim import DSSIM, SSIMBaseConfig


@dataclass
class MultiScaleDSSIMConfig(SSIMBaseConfig):
    """
    Configuration for the Multi-Scale DSSIM loss function.

    Attributes
    ----------
    name : str
        Name of the loss function.
    power_factors : list of float, optional
        Weighting factors for different scales (default is predefined values).
    level : int
        Number of scales to compute MS-DSSIM (default: 5).
    with_batches_averaging : bool
        Whether to average over batches (default: False).
    pool_strides : int
        Stride for downsampling (default: 2).
    pool_kernel : int
        Kernel size for downsampling (default: 2).

    """

    name: str = 'ms_dssim'
    power_factors: list[float] = None
    level: int = 5
    with_batches_averaging: bool = False
    pool_strides: int = 2
    pool_kernel: int = 2

    def __post_init__(self) -> None:
        """
        Validate and initialize default values for the configuration.

        Raises
        ------
        ValueError
            If `level` is greater than length of `power_factors`.

        """
        if self.power_factors is None:
            self.power_factors = [  # noqa: WPS601
                0.0448, 0.2856, 0.3001, 0.2363, 0.1333,
            ]
        max_level = len(self.power_factors)
        if self.level > max_level:
            msg = f'Level greater than {max_level} is not supported.'
            raise ValueError(msg)


class MultiScaleDSSIM(BaseLoss):
    """
    Multi-Scale DSSIM (MS-DSSIM) loss function.

    Attributes
    ----------
    config : MultiScaleDSSIMConfig
        Configuration of MultiScaleDSSIM.
    min_shape : int
        Minimum image shape required for computation.

    """

    config_type = MultiScaleDSSIMConfig

    def __init__(self, **kwargs) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__(**kwargs)
        scale = self.config.pool_strides ** (self.config.level - 1)
        self.min_shape = scale * self.config.filter_size

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """
        Return the MS-DSSIM loss.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth image tensor.
        y_pred : tf.Tensor
            Predicted image tensor.

        Returns
        -------
        tf.Tensor
            MS-DSSIM loss value.

        """
        ms_ssim = self.get_ms_ssim(y_true=y_true, y_pred=y_pred)
        return tf.divide(
            tf.convert_to_tensor(value=1, dtype=self.config.dtype) - ms_ssim,
            y=2,
        )

    def get_ms_ssim(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute the multiscale SSIM value.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth image tensor.
        y_pred : tf.Tensor
            Predicted image tensor.

        Returns
        -------
        tf.Tensor
            Multi-scale SSIM value per batch.

        """
        self._check_shapes(y_true=y_true, y_pred=y_pred)
        y_true, y_pred = self.cast_to_dtype(y_true=y_true, y_pred=y_pred)
        if self.config.max_val == 1:
            axis = (1, 2, 3)
            shape = (-1, 1, 1, 1)
            min_true = tf.reshape(
                tf.reduce_min(y_true, axis=axis),
                shape=shape,
            )
            max_true = tf.reshape(
                tf.reduce_max(y_true, axis=axis),
                shape=shape,
            )
            min_pred = tf.reshape(
                tf.reduce_min(y_pred, axis=axis),
                shape=shape,
            )
            max_pred = tf.reshape(
                tf.reduce_max(y_pred, axis=axis),
                shape=shape,
            )
            y_true = (y_true - min_true) / (max_true - min_true)
            y_pred = (y_pred - min_pred) / (max_pred - min_pred)
        weights = tf.constant(
            value=self.config.power_factors,
            dtype=self.config.dtype,
        )
        avg_pool_kwargs = {
            'ksize': self.config.pool_kernel,
            'strides': self.config.pool_strides,
            'padding': 'VALID',
        }

        ssim_maps = []
        cs_maps = []
        ssim_instance = self.get_dssim()
        for _ in range(self.config.level):
            ssim_out = ssim_instance.get_ssim(y_true=y_true, y_pred=y_pred)
            ssim_maps.append(self._relu_mean(ssim_out[0]))
            cs_maps.append(self._relu_mean(ssim_out[1]))

            filtered_im1 = tf.nn.avg_pool(y_true, **avg_pool_kwargs)
            filtered_im2 = tf.nn.avg_pool(y_pred, **avg_pool_kwargs)
            y_true = filtered_im1
            y_pred = filtered_im2

        ssim_tensor = self._stack_and_transpose(ssim_maps)
        cs_tensor = self._stack_and_transpose(cs_maps)

        weights = tf.expand_dims(weights, axis=-1)
        zero_level = 1e-20
        cs_values = tf.maximum(
            x=tf.constant(value=zero_level, dtype=self.config.dtype),
            y=cs_tensor[:, :self.config.level - 1],
        )
        luminance = tf.maximum(
            x=tf.constant(value=zero_level, dtype=self.config.dtype),
            y=ssim_tensor[:, self.config.level - 1],
        )
        weighted_cs_values = tf.math.pow(
            cs_values,
            weights[:self.config.level - 1],
        )
        weighted_luminance = tf.math.pow(
            luminance,
            weights[self.config.level - 1],
        )
        ms_ssim_per_channel = tf.reduce_prod(
            weighted_cs_values,
            axis=1,
        ) * weighted_luminance
        ms_ssim_per_batch = tf.reduce_mean(ms_ssim_per_channel, axis=-1)

        if self.config.with_batches_averaging:
            return tf.reduce_mean(ms_ssim_per_batch)
        return ms_ssim_per_batch

    def get_dssim(self) -> DSSIM:
        """
        Return a DSSIM instance with the configured parameters.

        Returns
        -------
        DSSIM
            Object for DSSIM calculation.

        """
        return DSSIM(
            dtype=self.config.dtype,
            max_val=self.config.max_val,
            filter_size=self.config.filter_size,
            filter_sigma=self.config.filter_sigma,
            k1=self.config.k1,
            k2=self.config.k2,
            return_cs_map=True,
            return_index_map=True,
            with_channels_averaging=False,
        )

    def _check_shapes(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> None:
        """
        Validate input tensor shapes.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth image tensor.
        y_pred : tf.Tensor
            Predicted image tensor.

        Raises
        ------
        ValueError
            If one of images is less than `min_shape`.

        """
        req_shape = (self.min_shape, self.min_shape)
        true_shape = y_true.shape[1:3]
        pred_shape = y_pred.shape[1:3]
        if any(s1 < s2 for s1, s2 in zip(true_shape, req_shape)):
            raise ValueError(
                f'True image {true_shape} is less than {req_shape}.',
            )
        if any(s1 < s2 for s1, s2 in zip(pred_shape, req_shape)):
            raise ValueError(
                f'Predicted image {pred_shape} is less than {req_shape}.',
            )

    @classmethod
    def _relu_mean(cls, tensor: tf.Tensor) -> tf.Tensor:
        """
        Calculate mean of image channel and apply ReLU.

        Parameters
        ----------
        tensor : tf.Tensor
            Input image.

        Returns
        -------
        tf.Tensor
            ReLU output.

        """
        mean = tf.reduce_mean(tensor, axis=(1, 2))
        # noinspection PyTypeChecker
        return tf.nn.relu(mean)

    @classmethod
    def _stack_and_transpose(cls, tensor_list: list[tf.Tensor]) -> tf.Tensor:
        """
        Stack list into tensor and transpose.

        Parameters
        ----------
        tensor_list : list of tf.Tensor
            List of tensors for stacking.

        Returns
        -------
        tf.Tensor
            Transposed tensor from the stacked list.

        """
        stacked = tf.stack(tensor_list, axis=0)
        return tf.transpose(stacked, perm=[1, 0, 2])
