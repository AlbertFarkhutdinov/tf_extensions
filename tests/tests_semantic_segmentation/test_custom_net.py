import pytest

from tf_extensions.semantic_segmentation import configs as cfg
from tf_extensions.semantic_segmentation.custom_layers import ConvolutionalBlock
from tf_extensions.semantic_segmentation.custom_net import CustomSegmentationNet

custom_net_properties = [
    (
        64,
        (3, 3),
        'relu',
        True,
        False,
        False,
        'glorot_uniform',
    ),
]


class TestCustomSegmentationNet:

    def test_init_without_args(self):
        model = CustomSegmentationNet()
        assert isinstance(model.config, cfg.CustomSegmentationNetConfig)

    @pytest.mark.parametrize(
        (
            'filters',
            'kernel',
            'activation',
            'bias',
            'bn',
            'dropout',
            'initializer',
        ),
        custom_net_properties,
    )
    def test_init(
        self,
        filters,
        kernel,
        activation,
        bias,
        bn,
        dropout,
        initializer,
    ):
        model = CustomSegmentationNet(
            config=cfg.CustomSegmentationNetConfig(
                initial_filters_number=filters,
                conv_block_config=cfg.ConvolutionalBlockConfig(
                    conv2d_config=cfg.Conv2DConfig(
                        kernel_size=kernel,
                        use_bias=bias,
                        kernel_initializer=initializer,
                    ),
                    activation=activation,
                    with_bn=bn,
                    with_dropout=dropout,
                ),
            ),
        )
        assert model.config.initial_filters_number == filters
        assert model.config.conv_block_config.layers_number == 2
        assert model.config.conv_block_config.activation == activation
        assert model.config.conv_block_config.with_bn == bn
        assert model.config.conv_block_config.with_dropout == dropout
        conv2d_config = model.config.conv_block_config.conv2d_config
        assert conv2d_config.kernel_size == kernel
        assert conv2d_config.padding == 'same'
        assert conv2d_config.use_bias == bias
        assert conv2d_config.kernel_initializer == initializer

    @pytest.mark.parametrize(
        ('filters', 'kernel_size'),
        [
            (64, (2, 2)),
            (64, (4, 4)),
        ],
    )
    def test_init_fail(self, filters, kernel_size):
        with pytest.raises(
            ValueError,
            match='Odd `kernel_size` is recommended.',
        ):
            CustomSegmentationNet(
                config=cfg.CustomSegmentationNetConfig(
                    initial_filters_number=filters,
                    conv_block_config=cfg.ConvolutionalBlockConfig(
                        conv2d_config=cfg.Conv2DConfig(
                            kernel_size=kernel_size,
                        ),
                    ),
                ),
            )

    @pytest.mark.parametrize(
        (
            'input_shape',
            'filters',
            'kernel',
            'activation',
            'bias',
            'bn',
            'dropout',
            'initializer',
        ),
        [((128, 128, 1), *item) for item in custom_net_properties],
    )
    def test_build_graph(
        self,
        input_shape,
        filters,
        kernel,
        activation,
        bias,
        bn,
        dropout,
        initializer,
    ):
        model = CustomSegmentationNet(
            config=cfg.CustomSegmentationNetConfig(
                initial_filters_number=filters,
                conv_block_config=cfg.ConvolutionalBlockConfig(
                    conv2d_config=cfg.Conv2DConfig(
                        kernel_size=kernel,
                        use_bias=bias,
                        kernel_initializer=initializer,
                    ),
                    activation=activation,
                    with_bn=bn,
                    with_dropout=dropout,
                ),
            ),
        )
        graph = model.build_graph(input_shape=input_shape)
        assert graph.inputs[0].shape == (None, *input_shape)
        assert graph.outputs[0].shape == (None, *input_shape)

    # @pytest.mark.parametrize(
    #     (
    #         'input_shape',
    #         'filters',
    #         'kernel',
    #         'activation',
    #         'bias',
    #         'bn',
    #         'dropout',
    #         'initializer',
    #     ),
    #     [((128, 128, 1), *item) for item in custom_net_properties],
    # )
    # def test_plot(
    #     self,
    #     input_shape,
    #     filters,
    #     kernel,
    #     activation,
    #     bias,
    #     bn,
    #     dropout,
    #     initializer,
    # ):
    #     model = CustomSegmentationNet(
    #         config=CustomSegmentationNetConfig(
    #             initial_filters_number=filters,
    #             conv_block_config=ConvolutionalBlockConfig(
    #                 conv2d_config=Conv2DConfig(
    #                     kernel_size=kernel,
    #                     use_bias=bias,
    #                     kernel_initializer=initializer,
    #                 ),
    #                 activation=activation,
    #                 with_bn=bn,
    #                 with_dropout=dropout,
    #             ),
    #         ),
    #     )
    #     model.plot(input_shape=input_shape)
    #     path = Path(f'{model.__class__.__name__}.png')
    #     assert path.exists()
    #     path.unlink(missing_ok=True)

    @pytest.mark.parametrize(
        (
            'filter_scale',
            'conv_kernel',
            'filters',
            'kernel',
            'act',
            'bias',
            'bn',
            'dropout',
            'init',
        ),
        [
            *[(2, None, *item) for item in custom_net_properties],
            *[(2, (5, 5), *item) for item in custom_net_properties],
        ],
    )
    def test_get_convolutional_pair(
        self,
        filter_scale,
        conv_kernel,
        filters,
        kernel,
        act,
        bias,
        bn,
        dropout,
        init,
    ):
        model = CustomSegmentationNet(
            config=cfg.CustomSegmentationNetConfig(
                initial_filters_number=filters,
                conv_block_config=cfg.ConvolutionalBlockConfig(
                    conv2d_config=cfg.Conv2DConfig(
                        kernel_size=kernel,
                        use_bias=bias,
                        kernel_initializer=init,
                    ),
                    activation=act,
                    with_bn=bn,
                    with_dropout=dropout,
                ),
            ),
        )
        conv_block = model.get_convolutional_block(
            filter_scale=filter_scale,
            kernel_size=conv_kernel,
        )
        assert isinstance(conv_block, ConvolutionalBlock)
        assert conv_block.filters == filters * filter_scale
        assert conv_block.config.layers_number == 2
        assert conv_block.config.activation == act
        assert conv_block.config.with_bn == bn
        assert conv_block.config.with_dropout == dropout
        conv2d_config = conv_block.config.conv2d_config
        assert conv2d_config.kernel_size == conv_kernel or kernel
        assert conv2d_config.padding == 'same'
        assert conv2d_config.use_bias == bias
        assert conv2d_config.kernel_initializer == init
