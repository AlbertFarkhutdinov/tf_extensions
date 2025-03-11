"""The module contains LSTM model and LSTM cell."""
from typing import Optional, Union

import tensorflow as tf


class Gate:
    """The class for a LSTM gate."""

    def __init__(self, hidden_size: int) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        self.input_ = tf.keras.layers.Dense(hidden_size)
        self.hidden = tf.keras.layers.Dense(hidden_size, use_bias=False)

    def output(self, input_tensor: tf.Tensor, hidden: tf.Tensor) -> tf.Tensor:
        """Return sum of outputs from `input_` and `hidden` layers."""
        return self.input_(input_tensor) + self.hidden(hidden)


class LSTMCell(tf.keras.Model):
    """The class for a LSTM cell."""

    def __init__(self, hidden_size: int) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__()
        self.input_gate = Gate(hidden_size=hidden_size)
        self.forget_gate = Gate(hidden_size=hidden_size)
        self.new_info_gate = Gate(hidden_size=hidden_size)
        self.output_gate = Gate(hidden_size=hidden_size)

    def __call__(
        self,
        input_tensor: tf.Tensor,
        hidden: tf.Tensor,
        cell_state: tf.Tensor,
    ):
        """Call self as a function."""
        i_output = tf.nn.sigmoid(
            self.input_gate.output(input_tensor, hidden),
        )
        f_output = tf.nn.sigmoid(
            self.forget_gate.output(input_tensor, hidden),
        )
        o_output = tf.nn.sigmoid(
            self.output_gate.output(input_tensor, hidden),
        )
        n_output = tf.nn.tanh(
            self.new_info_gate.output(input_tensor, hidden),
        )
        cell_state = f_output * cell_state + i_output * n_output
        return o_output * tf.nn.tanh(cell_state), cell_state


class LSTM(tf.keras.Model):
    """The class for a LSTM model."""

    def __init__(self, hidden_size: int) -> None:
        """Initialize self. See help(type(self)) for accurate signature."""
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = LSTMCell(hidden_size)

    def call(
        self,
        inputs: tf.Tensor,
        training: Union[bool, tf.Tensor] = None,
        mask: Union[Optional[tf.Tensor], list[Optional[tf.Tensor]]] = None,
    ) -> tf.Tensor:
        """Call self as a function."""
        batch = inputs.shape[0]
        hidden = tf.zeros((batch, self.hidden_size))
        cell_state = tf.zeros((batch, self.hidden_size))
        hidden_all = []

        for ind in range(inputs.shape[1]):
            hidden, cell_state = self.lstm_cell(
                input_tensor=inputs[:, ind, :],
                hidden=hidden,
                cell_state=cell_state,
            )
            hidden_all.append(hidden)

        return tf.transpose(tf.stack(hidden_all), perm=[1, 0, 2])
