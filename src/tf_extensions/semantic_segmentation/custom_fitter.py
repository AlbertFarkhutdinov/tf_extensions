"""The module to manage training and loading of a segmentation model."""
import logging
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from tf_extensions.semantic_segmentation.custom_net import \
    CustomSegmentationNet
from tf_extensions.semantic_segmentation.models_file_system import \
    ModelsFileSystem

DEFAULT_BATCH_SIZE = 32
DEFAULT_VAL_SPLIT = 0
DEFAULT_MONITOR = 'val_loss'
DEFAULT_MIN_DELTA = 5e-5
DEFAULT_PATIENCE = 10


class CustomFitter:
    """
    A class for to manage training and loading of a segmentation model.

    Parameters
    ----------
    dirs : str
        Directories for storing model-related files.
    model : CustomSegmentationNet
        An instance of the segmentation model.

    """

    def __init__(
        self,
        *dirs,
        model: CustomSegmentationNet,
    ) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        self.file_system = ModelsFileSystem(*dirs, model=model)

    def load_model(
        self,
        loaded_epoch: int = None,
    ) -> tuple[Optional[CustomSegmentationNet], pd.DataFrame]:
        """
        Load a model and its training history.

        Parameters
        ----------
        loaded_epoch : int, optional
            The epoch number to load the model from.
            If None, the latest model is loaded.

        Returns
        -------
        tuple
            - The loaded model : CustomSegmentationNet, optional
            - The training history : pd.DataFrame

        """
        model = self.file_system.load_model(epoch=loaded_epoch)
        fitting_history = self.file_system.load_fitting_history()
        if loaded_epoch:
            fitting_history = fitting_history.loc[
                fitting_history['epoch'] < loaded_epoch,
            ]
        return model, fitting_history

    def fit_model(
        self,
        model: CustomSegmentationNet,
        samples: np.ndarray,
        labels: np.ndarray,
        with_early_stopping: bool = False,
        period: int = 5,
        **fitting_kwargs,
    ) -> tuple[CustomSegmentationNet, pd.DataFrame]:
        """
        Train the segmentation model.

        Parameters
        ----------
        model : CustomSegmentationNet
            The segmentation model to be trained.
        samples : np.ndarray
            Input training samples.
        labels : np.ndarray
            Ground truth labels corresponding to `samples`.
        with_early_stopping : bool, optional, default: False
            Whether to use early stopping during training.
        period : int, optional, default: 5
            The frequency (in epochs) to save model weights.

        Returns
        -------
        tuple
            - The loaded model : CustomSegmentationNet
            - The training history : pd.DataFrame

        """
        logging.info('Fitting a new model.')
        batch_size = fitting_kwargs.get('batch_size', DEFAULT_BATCH_SIZE)
        validation_split = fitting_kwargs.get(
            'validation_split',
            DEFAULT_VAL_SPLIT,
        )
        monitor = fitting_kwargs.pop('monitor', DEFAULT_MONITOR)
        min_delta = fitting_kwargs.pop('min_delta', DEFAULT_MIN_DELTA)
        patience = fitting_kwargs.pop('patience', DEFAULT_PATIENCE)
        callbacks = []
        if with_early_stopping:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=monitor,
                    min_delta=min_delta,
                    patience=patience,
                    verbose=1,
                    restore_best_weights=True,
                ),
            )
        save_freq = period * int(
            np.ceil(
                len(samples) * (1 - validation_split) / batch_size,
            ),
        )
        msg = 'Saving frequency: {0} batches.'.format(save_freq)
        logging.info(msg)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.file_system.models_path.joinpath(
                    'epochs',
                    'weights.{epoch:03d}.tf',
                ),
                monitor=monitor,
                verbose=1,
                save_weights_only=True,
                save_freq=save_freq,
            ),
        )
        callbacks.append(
            tf.keras.callbacks.CSVLogger(
                filename=self.file_system.history_path,
                separator=',',
                append=True,
            ),
        )
        model.fit(
            samples,
            labels,
            callbacks=callbacks,
            **fitting_kwargs,
        )
        fitting_history = self.file_system.load_fitting_history()
        if with_early_stopping:
            stopped_epoch = callbacks[0].stopped_epoch
            if stopped_epoch:
                best_epoch = stopped_epoch - callbacks[0].patience + 1
                fitting_history = fitting_history.loc[
                    fitting_history['epoch'] < best_epoch,
                ]
        self.file_system.save_model_weights()
        self.file_system.save_fitting_history(fitting_history=fitting_history)
        self.file_system.save_config(
            **fitting_kwargs,
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            with_early_stopping=with_early_stopping,
        )
        return model, fitting_history
