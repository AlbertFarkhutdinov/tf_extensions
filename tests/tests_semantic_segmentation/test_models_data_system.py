import shutil

import numpy as np
import pandas as pd

from tf_extensions.semantic_segmentation.models_file_system import (
    ROOT_PATH,
    ModelsFileSystem,
)
from tf_extensions.semantic_segmentation.u_net import UNet

u_net = UNet()


class TestModelsDataSystem:

    def test_init_without_args(self):
        mds = ModelsFileSystem(model=u_net)
        assert mds.models_path.exists()
        assert mds.models_path.is_dir()
        assert mds.models_path.name == u_net.__class__.__name__
        assert mds.models_path.parent.name == 'models'

    def test_init_in_folder(self):
        test_folder = 'test_folder'
        mds = ModelsFileSystem(test_folder, model=u_net)
        assert mds.models_path.exists()
        assert mds.models_path.is_dir()
        assert mds.models_path.name == test_folder
        assert mds.models_path.parents[0].name == u_net.__class__.__name__
        assert mds.models_path.parents[1].name == 'models'
        shutil.rmtree(
            ROOT_PATH / 'models' / u_net.__class__.__name__ / test_folder,
        )

    def test_init_with_name(self):
        test_name = 'test_name'
        mds = ModelsFileSystem(model=u_net, model_name=test_name)
        assert mds.models_path.exists()
        assert mds.models_path.is_dir()
        assert mds.models_path.name == test_name
        assert mds.models_path.parent.name == 'models'
        shutil.rmtree(ROOT_PATH / 'models' / test_name)

    def test_init_in_folder_with_name(self):
        test_folder = 'test_folder'
        test_name = 'test_name'
        mds = ModelsFileSystem(
            test_folder,
            model=u_net,
            model_name=test_name,
        )
        assert mds.models_path.exists()
        assert mds.models_path.is_dir()
        assert mds.models_path.name == test_folder
        assert mds.models_path.parents[0].name == test_name
        assert mds.models_path.parents[1].name == 'models'
        shutil.rmtree(ROOT_PATH / 'models' / test_name)

    def test_save_and_load_config(self):
        test_folder = 'test_folder'
        test_name = 'test_name'
        mds = ModelsFileSystem(
            test_folder,
            model=u_net,
            model_name=test_name,
        )
        epochs = 2
        expected = {
            'model': 'UNet',
            'config': mds.model.config.as_dict(),
            'fitting_kwargs': {'epochs': epochs},
        }
        mds.save_config(epochs=epochs)
        assert mds.models_path.joinpath('config.json').exists()
        loaded_config = mds.load_config()
        assert sorted(expected.keys()) == sorted(loaded_config.keys())
        assert expected['model'] == loaded_config['model']
        assert expected['fitting_kwargs'] == loaded_config['fitting_kwargs']
        for field_name, field_value in expected['config'].items():
            if field_name != 'conv_block_config':
                assert loaded_config['config'][field_name] == field_value

        conv_block_config = loaded_config['config']['conv_block_config']
        exp_conv_block_config = expected['config']['conv_block_config']
        for field_name, field_value in exp_conv_block_config.items():
            if field_name != 'conv2d_config':
                assert conv_block_config[field_name] == field_value

        conv2d_config = conv_block_config['conv2d_config']
        exp_conv2d_config = exp_conv_block_config['conv2d_config']
        for field_name, field_value in exp_conv2d_config.items():
            if isinstance(conv2d_config[field_name], list):
                assert tuple(conv2d_config[field_name]) == field_value
            else:
                assert conv2d_config[field_name] == field_value
        shutil.rmtree(ROOT_PATH / 'models' / test_name)
        assert mds.load_config() == {}

    def test_save_and_load_fitting_history(self):
        test_folder = 'test_folder'
        test_name = 'test_name'
        mds = ModelsFileSystem(
            test_folder,
            model=u_net,
            model_name=test_name,
        )
        fitting_history = pd.DataFrame({
            'epoch': [0, 1],
            'loss': [1, 0],
            'metric': [0, 1],
        })
        mds.save_fitting_history(fitting_history=fitting_history)
        assert mds.models_path.joinpath('training.csv').exists()
        loaded_history = mds.load_fitting_history()
        assert set(loaded_history.columns) == set(fitting_history.columns)
        for column in fitting_history.columns:
            assert np.allclose(
                loaded_history[column].values,
                fitting_history[column].values,
            )

        shutil.rmtree(ROOT_PATH / 'models' / test_name)
        loaded_history = mds.load_fitting_history()
        assert set(loaded_history.columns) == {'epoch', 'loss'}
        for column in loaded_history.columns:
            assert not loaded_history[column].values.size

    def test_save_and_load_weights(self):
        test_folder = 'test_folder'
        test_name = 'test_name'
        mds = ModelsFileSystem(
            test_folder,
            model=u_net,
            model_name=test_name,
        )
        saved_weights = mds.model.weights
        mds.save_model_weights()
        assert mds.models_path.joinpath('weights.tf.index').exists()
        assert mds.models_path.joinpath('checkpoint').exists()
        updated_model = mds.load_model()
        assert np.allclose(saved_weights, updated_model.weights)
        updated_model = mds.load_model(epoch=15)
        assert updated_model is None
        shutil.rmtree(ROOT_PATH / 'models' / test_name)
