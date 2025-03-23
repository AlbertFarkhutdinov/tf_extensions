"""Module contains base class for losses using VGG."""
from dataclasses import dataclass

import tensorflow as tf
from keras.applications import VGG19
from keras.models import Model

from tf_extensions.losses.base_loss import BaseLoss, BaseLossConfig


@dataclass
class VGGBaseConfig(BaseLossConfig):
    """
    Configuration class for VGG-based loss.

    Attributes
    ----------
    name : str
        Name of the loss function.
    layer_names : list of str, optional
        List of layer names to extract features from.
        Defaults to ['block5_conv2'] if None.
    batch_size : int, optional
        Batch size for feature extraction.

    """

    name: str = 'vgg_base'
    layer_names: list[str] = None
    batch_size: int = None

    def __post_init__(self) -> None:
        """Update config properties after initialization."""
        if self.layer_names is None:
            self.layer_names = ['block5_conv4']  # noqa: WPS601


class VGGBase(BaseLoss):
    """
    Base class for losses using VGG.

    Attributes
    ----------
    config : VGGBaseConfig
        Configuration of VGGBase.
    model : tf.keras.Model
        A feature extractor model using VGG19.

    """

    config_type = VGGBaseConfig

    def __init__(self, **kwargs) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__(**kwargs)
        layer_names = self.config.layer_names
        self.model = self._get_feature_extractor(layer_names=layer_names)

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """
        Return the loss between the true and predicted images.

        Parameters
        ----------
        y_true : array-like
            Ground truth images.
        y_pred : array-like
            Predicted images.

        Returns
        -------
        float
            The computed loss.

        """
        y_true, y_pred = self._preprocess_images(y_true=y_true, y_pred=y_pred)
        true_features, pred_features = self._get_features(
            y_true=y_true,
            y_pred=y_pred,
        )
        return self._compute_loss(
            true_features=true_features,
            pred_features=pred_features,
        )

    @classmethod
    def _get_feature_extractor(
        cls,
        layer_names: list[str],
    ) -> tf.keras.Model:
        """
        Return a feature extractor model using VGG19.

        Parameters
        ----------
        layer_names : list of str
            Names of the VGG19 layers to extract features from.

        Returns
        -------
        tf.keras.Model
            A feature extraction model with selected layer outputs.

        """
        vgg = VGG19(weights='imagenet', include_top=False)
        outputs = [
            vgg.get_layer(name).output
            for name in layer_names
        ]
        model = Model(inputs=vgg.input, outputs=outputs)
        model.trainable = False
        return model

    def _preprocess_images(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Preprocess input images before feature extraction.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth images.
        y_pred : tf.Tensor
            Predicted images.

        Returns
        -------
        tuple[tf.Tensor, tf.Tensor]
            Processed y_true and y_pred tensors.

        """
        y_true, y_pred = self.cast_to_dtype(y_true=y_true, y_pred=y_pred)
        y_true, y_pred = self.normalize_images(y_true=y_true, y_pred=y_pred)
        unit_tensor = tf.convert_to_tensor(value=1, dtype=self.config.dtype)
        y_true = (y_true + unit_tensor) / 2
        y_pred = (y_pred + unit_tensor) / 2
        return y_true, y_pred

    def _get_features(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tuple[list[tf.Tensor], list[tf.Tensor]]:
        """
        Extract features from images using VGG19.

        Parameters
        ----------
        y_true : tf.Tensor
            Ground truth images.
        y_pred : tf.Tensor
            Predicted images.

        Returns
        -------
        tuple[list[tf.Tensor], list[tf.Tensor]]
            Extracted features for true and predicted images.

        """
        if self.config.batch_size is None:
            true_features = self.model(y_true)
            pred_features = self.model(y_pred)
            if not isinstance(true_features, list):
                true_features = [true_features]
                pred_features = [pred_features]
        else:
            combined_features = self.model.predict(
                tf.concat([y_true, y_pred], axis=0),
                batch_size=self.config.batch_size,
            )
            if not isinstance(combined_features, list):
                combined_features = [combined_features]

            mid = tf.shape(y_true)[0]
            true_features = [feat[:mid] for feat in combined_features]
            pred_features = [feat[mid:] for feat in combined_features]
        return true_features, pred_features

    def _compute_loss(
        self,
        true_features: list[tf.Tensor],
        pred_features: list[tf.Tensor],
    ) -> tf.Tensor:
        """
        Return the perceptual loss between extracted features.

        Parameters
        ----------
        true_features : list of tf.Tensor
            Extracted features from ground truth images.
        pred_features : list of tf.Tensor
            Extracted features from predicted images.

        Returns
        -------
        tf.Tensor
            Computed loss value.

        """
        losses = []
        for true_feat, pred_feat in zip(true_features, pred_features):
            loss = tf.reduce_mean(
                tf.square(
                    tf.subtract(
                        tf.cast(true_feat, self.config.dtype),
                        tf.cast(pred_feat, self.config.dtype),
                    ),
                ),
                axis=[1, 2, 3],
            )
            losses.append(loss)
        return tf.reduce_sum(losses, axis=0)
