"""The module provides class for saving and loading TensorFlow/Keras models."""
import json
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import tensorflow as tf

ROOT_PATH = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_PATH / 'models'
FittingHistory = dict[str, list[float]]


class ModelsFileSystem:
    """
    A class for saving and loading TensorFlow/Keras models.

    Parameters
    ----------
    *dirs : str
        Additional directory names to organize model files.
    model : tf.keras.Model
        The model instance to manage.
    model_name : str, optional
        Custom name for the model directory.
        If it is None, the model class name is used.

    Attributes
    ----------
    root_path : Path
        The root directory of the project.
    models_path : Path
        A directory to store models.
    history_path : Path
        A directory to store training history.

    """

    def __init__(
        self,
        *dirs,
        model: tf.keras.Model,
        model_name: str = None,
    ) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        self.root_path = ROOT_PATH
        self.models_path = MODEL_PATH
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.model_name = model_name or self.model.__class__.__name__
        self.models_path = self.models_path.joinpath(
            self.model_name,
            *dirs,
        )
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.history_path = self.models_path.joinpath('training.csv')

    def save_config(self, **fitting_kwargs) -> None:
        """Save the model configuration and training parameters to a file."""
        path = self.models_path
        config = {
            'model': self.model.__class__.__name__,
            'fitting_kwargs': {
                kw_name: kw_value
                for kw_name, kw_value in fitting_kwargs.items()
                if kw_name != 'validation_data'
            },
        }
        if hasattr(self.model, 'config'):
            config['config'] = self.model.config.as_dict()
        with path.joinpath('config.json').open(mode='w') as config_file:
            # noinspection PyTypeChecker
            json.dump(config, config_file)
        msg = 'The model config is saved to "{0}".'.format(path)
        logging.info(msg)

    def load_config(self) -> dict[str, Any]:
        """
        Load the saved model configuration.

        Returns
        -------
        dict
            Model configuration dictionary.

        """
        path = self.models_path
        try:
            with open(path / 'config.json') as config_file:
                config = json.load(config_file)
        except FileNotFoundError:
            msg = 'The model config is not found in "{0}".'.format(path)
            logging.warning(msg)
            config = {}
        return config

    def save_fitting_history(
        self,
        fitting_history: pd.DataFrame,
    ) -> None:
        """
        Save the model training history to a CSV file.

        Parameters
        ----------
        fitting_history : pd.DataFrame
            Training history (epochs, losses and metrics).

        """
        path = self.history_path
        fitting_history.to_csv(path, index=False)
        msg = 'The fitting history is saved to "{0}".'.format(path)
        logging.info(msg)

    def load_fitting_history(self) -> pd.DataFrame:
        """
        Load the training history from a CSV file.

        Returns
        -------
        pd.DataFrame
            Training history (epochs, losses and metrics).
            If file is not found,
            empty dataframe with columns `epoch` and `loss` is returned.

        """
        path = self.history_path
        try:
            fitting_history = pd.read_csv(path)
        except FileNotFoundError:
            msg = 'The fitting history is not found in "{0}".'.format(path)
            logging.warning(msg)
            fitting_history = pd.DataFrame({}, columns=['epoch', 'loss'])
        return fitting_history

    def save_model_weights(self) -> None:
        """Save the model weights to a file."""
        path = self.models_path
        self.model.save_weights(path / 'weights.tf')
        msg = 'The model is saved to "{0}".'.format(path)
        logging.info(msg)

    def load_model(
        self,
        epoch: int = None,
    ) -> Optional[tf.keras.Model]:
        """
        Load the model weights from a file.

        Parameters
        ----------
        epoch : int, optional
            If specified, loads weights from a specific training epoch.

        Returns
        -------
        tf.keras.Model, optional
            The model with loaded weights, or None if not found.

        """
        if epoch:
            file_name = 'weights.{0:03d}.tf'.format(epoch)
            path = self.models_path / 'epochs' / file_name
        else:
            path = self.models_path / 'weights.tf'
        try:
            self.model.load_weights(path).expect_partial()
            msg = 'The model weights are loaded from "{0}".'.format(path)
            logging.info(msg)
            return self.model
        except tf.errors.NotFoundError:
            msg = 'The saved model is not found in "{0}".'.format(path)
            logging.warning(msg)
            return None
