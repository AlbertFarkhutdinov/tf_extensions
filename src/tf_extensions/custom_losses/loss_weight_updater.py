import logging

import tensorflow as tf

from tf_extensions.custom_losses.combined_loss import CombinedLoss


class LossWeightUpdater(tf.keras.callbacks.Callback):

    def __init__(
        self,
        combined_loss: CombinedLoss,
    ) -> None:
        super().__init__()
        self.combined_loss = combined_loss

    def on_epoch_begin(
        self,
        epoch: int,
        logs: dict = None,
    ) -> None:
        """
        Update `combined_loss` at the start of an epoch.

        Parameters
        ----------
        epoch : int
            Index of epoch.
        logs : dict, optional
            Currently no data is passed to this argument for this method
            but that may change in the future.

        """
        self.combined_loss.update_weights(epoch=epoch)
        msg = 'Epoch {0}. Loss weights: {1}'.format(
            epoch,
            self.combined_loss.config.weights,
        )
        logging.info(msg)
