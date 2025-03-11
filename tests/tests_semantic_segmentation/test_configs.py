from tf_extensions.semantic_segmentation import configs as cfg

def_conv2d = {
    'kernel_size': (3, 3),
    'padding': 'same',
    'use_bias': True,
    'kernel_initializer': 'glorot_uniform',
}
def_conv_block = {
    'conv2d_config': def_conv2d,
    'drop_rate': 0.5,
    'layers_number': 2,
    'activation': 'relu',
    'with_bn': False,
    'with_dropout': False,
    'with_skipped': False,
}
def_custom_net = {
    'conv_block_config': def_conv_block,
    'initial_filters_number': 16,
    'max_filters_number': None,
}
def_seg_net = {
    'conv_block_config': def_conv_block,
    'initial_filters_number': 16,
    'max_filters_number': None,
    'path_length': 4,
    'pooling': 2,
}
def_u_net = {
    'conv_block_config': def_conv_block,
    'initial_filters_number': 16,
    'max_filters_number': None,
    'path_length': 4,
    'pooling': 2,
    'without_reducing_filters': False,
    'is_partial_reducing': True,
    'out_residual_blocks_number': 0,
    'is_skipped_with_concat': True,
    'first_kernel_size': None,
    'vector_length': None,
    'is_binary_classification': False,
    'with_attention': False,
    'with_variable_kernel': False,
    'first_blocks_without_dropout': 0,
}


class TestConv2DConfig:

    def test_init(self):
        config = cfg.Conv2DConfig()
        assert config.kernel_size == def_conv2d['kernel_size']
        assert config.padding == def_conv2d['padding']
        assert config.use_bias == def_conv2d['use_bias']
        assert config.kernel_initializer == def_conv2d['kernel_initializer']

    def test_as_dict(self):
        config = cfg.Conv2DConfig()
        assert config.as_dict() == def_conv2d

    def test_from_dict(self):
        config = cfg.Conv2DConfig()
        assert config.from_dict(properties=def_conv2d) == config

    def test_config_name(self):
        config = cfg.Conv2DConfig(
            kernel_size=(5, 5),
            padding='valid',
            use_bias=False,
            kernel_initializer='he_normal',
        )
        config_name = config.get_config_name()
        assert config_name == 'kernel5x5_pad_valid_without_bias_init_he_normal'


class TestConvolutionalBlockConfig:

    def test_init(self):
        config = cfg.ConvolutionalBlockConfig()
        assert config.conv2d_config == cfg.Conv2DConfig()
        assert config.layers_number == def_conv_block['layers_number']
        assert config.activation == def_conv_block['activation']
        assert config.with_bn == def_conv_block['with_bn']
        assert config.with_dropout == def_conv_block['with_dropout']

    def test_as_dict(self):
        config = cfg.ConvolutionalBlockConfig()
        assert config.as_dict() == def_conv_block

    def test_from_dict(self):
        config = cfg.ConvolutionalBlockConfig()
        assert config.from_dict(properties=def_conv_block) == config

    def test_config_name(self):
        conv2d_config = cfg.Conv2DConfig(kernel_size=(5, 5))
        config = cfg.ConvolutionalBlockConfig(
            conv2d_config=conv2d_config,
            layers_number=3,
            with_skipped=True,
            with_bn=True,
            with_dropout=True,
            drop_rate=0.3,
        )
        config_name = config.get_config_name()
        assert config_name == 'relu3_residual_bn_drop30_kernel5x5'


class TestCustomSegmentationNetConfig:

    def test_init(self):
        config = cfg.CustomSegmentationNetConfig()
        assert config.conv_block_config == cfg.ConvolutionalBlockConfig()
        filters_number = config.initial_filters_number
        assert filters_number == def_custom_net['initial_filters_number']

    def test_as_dict(self):
        config = cfg.CustomSegmentationNetConfig()
        assert config.as_dict() == def_custom_net

    def test_from_dict(self):
        config = cfg.CustomSegmentationNetConfig()
        assert config.from_dict(properties=def_custom_net) == config

    def test_config_name(self):
        conv_block_config = cfg.ConvolutionalBlockConfig()
        custom_seg_net_config = cfg.CustomSegmentationNetConfig(
            conv_block_config=conv_block_config,
            initial_filters_number=32,
            max_filters_number=64,
        )
        config_name = custom_seg_net_config.get_config_name()
        assert config_name == 'input_neurons32_max_neurons64_relu2_kernel3x3'


class TestSegNetConfig:

    def test_init(self):
        config = cfg.SegNetConfig()
        assert config.conv_block_config == cfg.ConvolutionalBlockConfig()
        filters_number = config.initial_filters_number
        assert filters_number == def_seg_net['initial_filters_number']
        assert config.path_length == def_seg_net['path_length']
        assert config.pooling == def_seg_net['pooling']

    def test_as_dict(self):
        config = cfg.SegNetConfig()
        assert config.as_dict() == def_seg_net

    def test_from_dict(self):
        config = cfg.SegNetConfig()
        assert config.from_dict(properties=def_seg_net) == config

    def test_config_name(self):
        seg_net_config = cfg.SegNetConfig(
            path_length=5,
            pooling=3,
        )
        config_name = seg_net_config.get_config_name()
        assert config_name == (
            'encoder5_pooling3_input_neurons16_relu2_kernel3x3'
        )


class TestUNetConfig:

    def test_init(self):
        config = cfg.UNetConfig()
        assert config.conv_block_config == cfg.ConvolutionalBlockConfig()
        filters_number = config.initial_filters_number
        wrf = config.without_reducing_filters
        out = config.out_residual_blocks_number
        skip = config.is_skipped_with_concat
        assert filters_number == def_u_net['initial_filters_number']
        assert config.path_length == def_u_net['path_length']
        assert config.pooling == def_u_net['pooling']
        assert wrf == def_u_net['without_reducing_filters']
        assert out == def_u_net['out_residual_blocks_number']
        assert skip == def_u_net['is_skipped_with_concat']
        assert config.first_kernel_size == def_u_net['first_kernel_size']
        assert config.vector_length == def_u_net['vector_length']

    def test_as_dict(self):
        config = cfg.UNetConfig()
        assert config.as_dict() == def_u_net

    def test_from_dict(self):
        config = cfg.UNetConfig()
        assert config.from_dict(properties=def_u_net) == config

    def test_config_name(self):
        unet_config = cfg.UNetConfig(
            with_attention=True,
            without_reducing_filters=True,
            is_partial_reducing=False,
            first_blocks_without_dropout=2,
            out_residual_blocks_number=1,
            first_kernel_size=(7, 7),
            vector_length=128,
        )
        config_name = unet_config.get_config_name()
        assert config_name == ''.join(
            [
                'attention_',
                'without_reducing_filters_',
                '2without_drop_',
                'out_res1concat_',
                'first_kernel7x7_',
                'vector_length128_',
                'encoder4_',
                'input_neurons16_',
                'relu2_',
                'kernel3x3',
            ]
        )
