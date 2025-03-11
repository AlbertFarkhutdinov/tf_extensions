from tf_extensions.losses.adaptive_nmae import AdaptiveNMAE
from tf_extensions.losses.dists import DISTS
from tf_extensions.losses.dssim import DSSIM
from tf_extensions.losses.fft import FFTLoss
from tf_extensions.losses.ms_dssim import MultiScaleDSSIM
from tf_extensions.losses.soft_dice import SoftDiceLoss
from tf_extensions.losses.vgg import VGGLoss

supported_losses = {
    'AdaptiveNMAE': AdaptiveNMAE,
    'DISTS': DISTS,
    'DSSIM': DSSIM,
    'FFTLoss': FFTLoss,
    'MultiScaleDSSIM': MultiScaleDSSIM,
    'SoftDiceLoss': SoftDiceLoss,
    'VGGLoss': VGGLoss,
}
