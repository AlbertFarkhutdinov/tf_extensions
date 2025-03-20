"""The package contains custom Tensorflow losses."""
from tf_extensions.losses.combined_loss import CombinedLoss
from tf_extensions.losses.dists import DISTS
from tf_extensions.losses.dssim import DSSIM
from tf_extensions.losses.fft import FFTLoss
from tf_extensions.losses.ms_dssim import MultiScaleDSSIM
from tf_extensions.losses.multiscale_loss import MultiScaleLoss
from tf_extensions.losses.soft_dice import SoftDiceLoss
from tf_extensions.losses.vgg import VGGLoss
from tf_extensions.losses.vgg_base import VGGBase

__all__ = [
    'CombinedLoss',
    'DISTS',
    'DSSIM',
    'FFTLoss',
    'MultiScaleDSSIM',
    'MultiScaleLoss',
    'SoftDiceLoss',
    'VGGBase',
    'VGGLoss',
]
