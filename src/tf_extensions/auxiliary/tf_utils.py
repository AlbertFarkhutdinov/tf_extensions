"""The module provides helper functions for tf_extensions."""
import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


def set_memory_growth() -> str:
    """
    Enable memory growth for GPUs.

    Returns
    -------
    str
        Information about available devices.

    """
    gpu_list = tf.config.list_physical_devices('GPU')
    gpu_number = len(gpu_list)
    if gpu_number:
        try:
            for gpu in gpu_list:
                tf.config.experimental.set_memory_growth(
                    device=gpu,
                    enable=True,
                )
        except RuntimeError as exc:
            msg = str(exc)
            logger.exception(msg)
        logical_gpu_list = tf.config.experimental.list_logical_devices('GPU')
    else:
        logical_gpu_list = []
    return ', '.join(
        [
            f'{gpu_number} Physical GPUs',
            f'{len(logical_gpu_list)} Logical GPUs',  # noqa: WPS237
        ],
    )
