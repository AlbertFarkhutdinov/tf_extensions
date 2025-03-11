"""The module provides helper functions for semantic segmentation."""
import logging

import tensorflow as tf


def set_memory_growth() -> str:
    """
    Enable memory growth for GPUs.

    Returns
    -------
    str
        Information about available devices.

    """
    gpu_list = tf.config.list_physical_devices('GPU')
    if gpu_list:
        try:
            for gpu in gpu_list:
                tf.config.experimental.set_memory_growth(
                    device=gpu,
                    enable=True,
                )
        except RuntimeError as exc:
            logging.error(exc)
        logical_gpu_list = tf.config.experimental.list_logical_devices('GPU')
    else:
        logical_gpu_list = []
    return '{physical} Physical GPUs, {logical} Logical GPUs'.format(
        physical=len(gpu_list),
        logical=len(logical_gpu_list),
    )
