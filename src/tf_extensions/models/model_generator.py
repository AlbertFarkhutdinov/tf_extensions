from dataclasses import dataclass
from pathlib import Path
from typing import Union

import tensorflow as tf
import yaml
from keras import Model, layers

from tf_extensions.auxiliary.base_config import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    activation: str = 'relu'
    dropout_rate: float = None
    dropout_rate_residual: float = None
    initial_filters: int = 64
    residual_filters: int = None
    kernel_size: tuple[int, int] = (3, 3)
    max_filters: int = None
    model_depth: int = 3
    padding: str = 'same'
    residual_blocks_number: int = 3
    strides: tuple[int, int] = (2, 2)
    residual_strides: tuple[int, int] = (2, 2)
    use_bias: bool = False
    use_bias_residual: bool = True
    with_bn: bool = True
    with_max_pooling: bool = False
    with_multiple_outputs: bool = False
    with_output_activation: bool = False
    with_transpose: bool = False
    is_inverse: bool = False


class ModelGenerator:

    def __init__(self, **kwargs) -> None:
        self.config = ModelConfig.from_kwargs(**kwargs)

    def save_config(
        self,
        config_path: Path,
        **kwargs,
    ) -> None:
        with config_path.with_suffix('.yaml').open(mode='w') as info_file:
            config_dict = self.config.as_dict()
            config_dict = {**config_dict, **kwargs}
            yaml.safe_dump(config_dict, info_file)

    def get_conv_blocks(
        self,
        inputs: tf.Tensor,
        filters: int,
        block_id: Union[str, int],
        layers_number: int = 2,
        use_bias: bool = None,
        dropout_rate: float = None,
    ) -> tf.Tensor:
        if use_bias is None:
            use_bias = self.config.use_bias
        if dropout_rate is None:
            dropout_rate = self.config.dropout_rate
        if self.config.max_filters is not None:
            filters = min(filters, self.config.max_filters)
        outputs = inputs
        for layer_id in range(layers_number):
            postfix = '{0}_{1}'.format(block_id, layer_id)
            outputs = layers.Conv2D(
                filters=filters,
                name='conv{0}'.format(postfix),
                use_bias=use_bias,
                kernel_size=self.config.kernel_size,
                padding=self.config.padding,
            )(outputs)
            if self.config.with_bn:
                outputs = layers.BatchNormalization(
                    name='bn{0}'.format(postfix),
                )(outputs)
            outputs = layers.Activation(
                activation=self.config.activation,
                name='{0}{1}'.format(self.config.activation, postfix),
            )(outputs)
            if dropout_rate is not None:
                outputs = layers.SpatialDropout2D(
                    rate=dropout_rate,
                    name='dropout{0}'.format(postfix),
                )(outputs)
        return outputs

    def get_encoder_block(
        self,
        inputs: tf.Tensor,
        filters: int,
        block_id: Union[str, int],
    ) -> tf.Tensor:
        max_filters = self.config.max_filters
        is_inverse = self.config.is_inverse
        if max_filters is not None:
            filters = min(filters, max_filters)
            pooling_filters = min(filters, max_filters)
        elif is_inverse:
            pooling_filters = filters * 2
        else:
            pooling_filters = filters // 2
        outputs = inputs
        strides = self.config.strides
        if self.config.with_max_pooling:
            outputs = layers.MaxPooling2D(
                pool_size=strides,
                name='max_pool{0}'.format(block_id),
            )(outputs)
        else:
            outputs = layers.Conv2D(
                filters=pooling_filters,
                strides=strides,
                name='encoder_conv{0}'.format(block_id),
                kernel_size=self.config.kernel_size,
                padding=self.config.padding,
                use_bias=self.config.use_bias,
            )(outputs)
        return self.get_conv_blocks(
            inputs=outputs,
            filters=filters,
            block_id=block_id,
        )

    def get_decoder_block(
        self,
        inputs: list[tf.Tensor],
        filters: int,
        block_id: Union[str, int],
    ) -> tf.Tensor:
        if self.config.max_filters is not None:
            filters = min(filters, self.config.max_filters)
            up_filters = min(filters, self.config.max_filters)
        elif self.config.is_inverse:
            up_filters = filters // 2
        else:
            up_filters = filters * 2
        strides = self.config.strides
        outputs, encoder_outputs = inputs
        if self.config.with_transpose:
            outputs = layers.Conv2DTranspose(
                filters=up_filters,
                strides=strides,
                name='transpose{0}'.format(block_id),
                use_bias=self.config.use_bias,
            )(outputs)
        else:
            outputs = layers.UpSampling2D(
                size=strides,
                interpolation='bilinear',
                name='up_sampling{0}'.format(block_id),
            )(outputs)
        outputs = layers.Conv2D(
            filters=up_filters,
            name='decoder_conv{0}'.format(block_id),
            use_bias=self.config.use_bias,
            kernel_size=self.config.kernel_size,
            padding=self.config.padding,
        )(outputs)
        outputs = layers.Concatenate(
            name='concat{0}'.format(block_id),
        )([outputs, encoder_outputs])
        return self.get_conv_blocks(
            inputs=outputs,
            filters=filters,
            block_id=block_id,
            layers_number=2,
        )

    def get_residual_block(
        self,
        inputs: tf.Tensor,
        filters: int,
        block_id: Union[str, int],
    ) -> tf.Tensor:
        max_filters = self.config.max_filters
        if max_filters is not None:
            filters = min(filters, max_filters)
        outputs = self.get_conv_blocks(
            inputs=inputs,
            filters=filters,
            block_id=block_id,
            use_bias=self.config.use_bias_residual,
            dropout_rate=self.config.dropout_rate_residual,
        )
        return layers.Add(
            name='add{0}'.format(block_id),
        )([inputs, outputs])

    def get_output_blocks(
        self,
        inputs: tf.Tensor,
        filters: int,
        block_id: Union[str, int],
        residual_blocks_number: int,
        latent_dim: tuple[int, ...] = (64, 64, 1),
    ) -> tf.Tensor:
        max_filters = self.config.max_filters
        if max_filters is not None:
            filters = min(filters, max_filters)
        outputs = inputs
        residual_strides = self.config.residual_strides
        if residual_strides != (1, 1):
            outputs = layers.UpSampling2D(
                size=residual_strides,
                interpolation='bilinear',
                name='up_sampling_residual{0}'.format(block_id),
            )(outputs)
        outputs = layers.Conv2D(
            filters=filters,
            name='conv_residual{0}'.format(block_id),
            use_bias=self.config.use_bias,
            kernel_size=self.config.kernel_size,
            padding=self.config.padding,
        )(outputs)
        for res_id in range(residual_blocks_number):
            outputs = self.get_residual_block(
                inputs=outputs,
                filters=filters,
                block_id='{0}_{1}'.format(block_id, res_id),
            )

        outputs = layers.Conv2D(
            filters=latent_dim[-1],
            kernel_size=(1, 1),
            padding='same',
            use_bias=self.config.use_bias,
            name='conv_out{0}'.format(block_id),
        )(outputs)
        if self.config.with_output_activation:
            outputs = layers.Activation(
                activation='tanh',
                name='g_out{0}'.format(block_id),
            )(outputs)
        return outputs

    def get_encoder_outputs_list(
        self,
        inputs: tf.Tensor,
    ) -> list[tf.Tensor]:
        initial_filters = self.config.initial_filters
        model_depth = self.config.model_depth
        max_filters = self.config.max_filters
        is_inverse = self.config.is_inverse
        if is_inverse:
            filters = 2 ** model_depth * self.config.initial_filters
        else:
            filters = self.config.initial_filters
        if max_filters is not None:
            filters = min(max_filters, filters)
        outputs0 = layers.Conv2D(
            filters=filters,
            name='conv0',
            kernel_size=self.config.kernel_size,
            padding=self.config.padding,
            use_bias=self.config.use_bias,
        )(inputs)
        encoder_outputs_list = [
            self.get_conv_blocks(
                inputs=outputs0,
                block_id=1,
                filters=filters,
                layers_number=1,
            ),
        ]
        for layer_id in range(self.config.model_depth):
            if is_inverse:
                filters = 2 ** (model_depth - layer_id - 1) * initial_filters
            else:
                filters = 2 ** (layer_id + 1) * initial_filters
            encoder_outputs_list.append(
                self.get_encoder_block(
                    inputs=encoder_outputs_list[-1],
                    block_id=2 + layer_id,
                    filters=filters,
                ),
            )
        return encoder_outputs_list

    def get_decoder_outputs_list(
        self,
        encoder_outputs_list: list[tf.Tensor],
    ) -> list[tf.Tensor]:
        decoder_outputs_list = [encoder_outputs_list[-1]]
        initial_filters = self.config.initial_filters
        model_depth = self.config.model_depth
        for layer_id in range(model_depth):
            if self.config.is_inverse:
                filters = 2 ** (layer_id + 1) * initial_filters
            else:
                filters = 2 ** (model_depth - layer_id - 1) * initial_filters
            decoder_outputs = self.get_decoder_block(
                inputs=[
                    decoder_outputs_list[-1],
                    encoder_outputs_list[-2 - layer_id],
                ],
                block_id=2 + model_depth + layer_id,
                filters=filters,
            )
            decoder_outputs_list.append(decoder_outputs)
        return decoder_outputs_list

    def get_model(
        self,
        latent_dim: tuple[int, ...] = (64, 64, 1),
    ) -> Model:
        inputs = layers.Input(shape=latent_dim, name='input')
        encoder_outputs_list = self.get_encoder_outputs_list(
            inputs=inputs,
        )
        decoder_outputs_list = self.get_decoder_outputs_list(
            encoder_outputs_list=encoder_outputs_list,
        )

        residual_filters = self.config.residual_filters
        if residual_filters is None:
            residual_filters = self.config.initial_filters
        outputs_list = [
            self.get_output_blocks(
                inputs=decoder_outputs_list[-1],
                block_id=self.config.model_depth * 2 + 1,
                filters=residual_filters,
                residual_blocks_number=self.config.residual_blocks_number,
                latent_dim=latent_dim,
            ),
        ]
        if self.config.with_multiple_outputs:
            for layer_id in range(self.config.model_depth):
                outputs_list.append(
                    self.get_output_blocks(
                        inputs=decoder_outputs_list[-2 - layer_id],
                        block_id=self.config.model_depth * 2 - layer_id,
                        filters=(
                            2 ** (layer_id + 1) * self.config.initial_filters
                        ),
                        residual_blocks_number=0,
                        latent_dim=latent_dim,
                    ),
                )

            return Model(
                inputs=inputs,
                outputs=outputs_list,
            )
        return Model(inputs=inputs, outputs=outputs_list[0])
