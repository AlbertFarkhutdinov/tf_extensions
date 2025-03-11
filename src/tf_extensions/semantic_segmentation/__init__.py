"""The package provides custom layers and models for semantic segmentation."""
from tf_extensions.semantic_segmentation.aspp import ASPPNet
from tf_extensions.semantic_segmentation.configs import Conv2DConfig  # noqa: I001
from tf_extensions.semantic_segmentation.configs import ConvolutionalBlockConfig  # noqa: I001
from tf_extensions.semantic_segmentation.configs import CustomSegmentationNetConfig  # noqa: I001
from tf_extensions.semantic_segmentation.configs import SegNetConfig  # noqa: I001
from tf_extensions.semantic_segmentation.configs import UNetConfig  # noqa: I001; noqa: I001
from tf_extensions.semantic_segmentation.seg_net import SegNet  # noqa: I005
from tf_extensions.semantic_segmentation.ss_utils import set_memory_growth  # noqa: I005
from tf_extensions.semantic_segmentation.u_net import UNet  # noqa: I005

__all__ = [
    'ASPPNet',
    'Conv2DConfig',
    'ConvolutionalBlockConfig',
    'CustomSegmentationNetConfig',
    'SegNet',
    'SegNetConfig',
    'UNet',
    'UNetConfig',
    'set_memory_growth',
]
