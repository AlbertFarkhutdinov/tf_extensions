from typing import Optional, Union

import tensorflow as tf

TrainingType = Union[bool, tf.Tensor]
MaskType = Union[Optional[tf.Tensor], list[Optional[tf.Tensor]]]
