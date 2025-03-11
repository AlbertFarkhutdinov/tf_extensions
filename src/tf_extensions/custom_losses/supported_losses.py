from tf_extensions.custom_losses.adaptive_nmae import AdaptiveNMAE
from tf_extensions.custom_losses.custom_loss import CustomLoss
from tf_extensions.custom_losses.dists import DISTS
from tf_extensions.custom_losses.dssim import DSSIM
from tf_extensions.custom_losses.fft import FFTLoss
from tf_extensions.custom_losses.ms_dssim import MultiScaleDSSIM
from tf_extensions.custom_losses.soft_dice import SoftDiceLoss
from tf_extensions.custom_losses.vgg import VGGLoss

supported_losses = {
    'AdaptiveNMAE': AdaptiveNMAE,
    'CustomLoss': CustomLoss,
    'DISTS': DISTS,
    'DSSIM': DSSIM,
    'FFTLoss': FFTLoss,
    'MultiScaleDSSIM': MultiScaleDSSIM,
    'SoftDiceLoss': SoftDiceLoss,
    'VGGLoss': VGGLoss,
}
