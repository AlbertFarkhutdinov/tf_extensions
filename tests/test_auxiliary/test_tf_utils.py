from unittest import mock

import tensorflow as tf

from tf_extensions.auxiliary.tf_utils import set_memory_growth


def test_set_memory_growth_with_gpus():
    """Test `set_memory_growth` when GPUs are available."""
    mock_gpus = [mock.Mock()] * 2  # Simulate 2 physical GPUs
    mock_logical_gpus = [mock.Mock()] * 3  # Simulate 3 logical GPUs

    mock_physical_devices = mock.patch.object(
        tf.config,
        attribute='list_physical_devices',
        return_value=mock_gpus,
    )
    mock_logical_devices = mock.patch.object(
        tf.config.experimental,
        attribute='list_logical_devices',
        return_value=mock_logical_gpus,
    )
    mock_set_growth = mock.patch.object(
        tf.config.experimental,
        attribute='set_memory_growth',
    )
    with mock_physical_devices, mock_set_growth as msg, mock_logical_devices:
        result = set_memory_growth()
        assert result == '{phys} Physical GPUs, {log} Logical GPUs'.format(
            phys=len(mock_gpus),
            log=len(mock_logical_gpus),
        )
        assert msg.call_count == len(mock_gpus)


def test_set_memory_growth_no_gpus():
    """Test `set_memory_growth` when no GPUs are available."""
    mock_physical_devices = mock.patch.object(
        tf.config,
        attribute='list_physical_devices',
        return_value=[],
    )
    mock_logical_devices = mock.patch.object(
        tf.config.experimental,
        attribute='list_logical_devices',
        return_value=[],
    )
    with mock_physical_devices, mock_logical_devices:
        result = set_memory_growth()
        assert result == '0 Physical GPUs, 0 Logical GPUs'


def test_set_memory_growth_runtime_error(caplog):
    """Test `set_memory_growth` when RuntimeError occurs."""
    mock_gpus = [mock.Mock()]  # Simulate 1 physical GPU
    error_message = 'Test error'

    mock_physical_devices = mock.patch.object(
        tf.config,
        attribute='list_physical_devices',
        return_value=mock_gpus,
    )
    mock_logical_devices = mock.patch.object(
        tf.config.experimental,
        attribute='list_logical_devices',
        return_value=mock_gpus,
    )
    mock_set_growth = mock.patch.object(
        tf.config.experimental,
        attribute='set_memory_growth',
        side_effect=RuntimeError(error_message),
    )

    with mock_physical_devices, mock_set_growth, mock_logical_devices:
        result = set_memory_growth()

        assert result == "1 Physical GPUs, 1 Logical GPUs"
        assert error_message in caplog.text  # Check that the error was logged
