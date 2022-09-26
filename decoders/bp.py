"""
GNBP RNN Cell 

Brief: Description of the GNBP RNN Cell (which can be used for both conventional BP or GNBP  decoding)

Copyright (c) 2022 Orange

Author: Guillaume Larue <guillaume.larue@orange.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice (including the next paragraph) shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
"""

import tensorflow as tf
from activations import (
    AtanhTaylorApproxActivation,
    AtanhActivation,
)
from regularizers import L2WeightRegularizer


class GatedNeuralBeliefPropagationRNNCell(tf.keras.layers.Layer):
    def __init__(
        self,
        n_variable_nodes,
        n_check_nodes,
        trainable=True,
        **kwargs,
    ):

        super(GatedNeuralBeliefPropagationRNNCell, self).__init__(**kwargs)

        self.n_variable_nodes = n_variable_nodes
        self.n_check_nodes = n_check_nodes
        self.trainable = trainable
        self.state_size = tf.TensorShape([n_variable_nodes * n_check_nodes])
        self.output_size = tf.TensorShape([n_variable_nodes])
        self.atanh_activation_layer_eval = AtanhActivation()
        self.atanh_activation_layer_training = AtanhTaylorApproxActivation(order=21)

    def build(self, input_shape):

        self.factor_graph_weights_sum = self.add_weight(
            shape=(self.n_check_nodes * self.n_variable_nodes,),
            initializer=tf.keras.initializers.Constant(1.0),
            regularizer=L2WeightRegularizer(alpha=5e-2, mean=1),
            name="factor_graph_weights_sum",
            trainable=self.trainable,
        )

        self.factor_graph_weights_prod = tf.ones(
            shape=(self.n_check_nodes * self.n_variable_nodes),
            name="factor_graph_weights_prod",
        )

        self.factor_graph_weights_out = self.add_weight(
            shape=(self.n_check_nodes * self.n_variable_nodes,),
            initializer=tf.keras.initializers.Constant(1.0),
            regularizer=L2WeightRegularizer(alpha=5e-2, mean=1),
            name="factor_graph_weights_out",
            trainable=self.trainable,
        )

        self.input_weights = tf.ones(
            shape=(self.n_variable_nodes), name="SP_input_weights"
        )
        self.input_weights_out = tf.ones(
            shape=(self.n_variable_nodes), name="SP_input_weights_out"
        )

    def call(self, inputs, states, constants=None, training=False):
        ## constants
        llr = inputs

        H = constants
        # H = differentiable_step_function(H)

        factor_graph_gate = tf.reshape(
            H,
            [
                self.n_check_nodes * self.n_variable_nodes,
            ],
        )

        sum_gate = factor_graph_gate
        sum_weights = self.factor_graph_weights_sum

        reshaped_factor_graph_gate = factor_graph_gate
        reshaped_factor_graph_gate = tf.reshape(
            reshaped_factor_graph_gate, [self.n_check_nodes, self.n_variable_nodes]
        )
        reshaped_factor_graph_gate = tf.reverse(reshaped_factor_graph_gate, axis=[-2])
        reshaped_factor_graph_gate = tf.reshape(
            reshaped_factor_graph_gate, [self.n_check_nodes * self.n_variable_nodes]
        )
        prod_gate_weights = reshaped_factor_graph_gate
        prod_gate_bias = 1 - reshaped_factor_graph_gate
        prod_weights = self.factor_graph_weights_prod

        out_gate = factor_graph_gate
        out_weights = self.factor_graph_weights_out

        ############################ SUM ITERATION ############################
        normalized_inputs = llr
        llr = tf.nest.flatten(normalized_inputs)
        weighted_inputs = tf.multiply(llr, self.input_weights)
        reshaped_inputs = tf.reshape(weighted_inputs, [-1, 1, self.n_variable_nodes])
        repeated_inputs = tf.repeat(reshaped_inputs, (self.n_check_nodes), axis=-2)

        # STATES PRE-PROCESSING
        # weight states
        weighted_states = tf.multiply(states, sum_weights)

        # Gate states
        gated_states = tf.multiply(weighted_states, sum_gate)

        reshaped_states = tf.reshape(
            gated_states, [-1, self.n_check_nodes, self.n_variable_nodes]
        )
        repeated_states = tf.repeat(reshaped_states, (self.n_check_nodes - 1), axis=-2)

        # CONCATENATE
        x = tf.concat([repeated_states, repeated_inputs], axis=-2)

        # RESHAPE
        x = tf.reshape(
            x, [-1, self.n_check_nodes, self.n_check_nodes * self.n_variable_nodes]
        )

        # REDUCE SUM
        x = tf.reduce_sum(x, axis=-2)

        ############################ PRODUCT ITERATION ############################
        # Weights
        x = tf.multiply(x, prod_weights)

        # TANH[sum(x)/2]
        x = tf.tanh(x / 2)

        # Gate
        x = tf.multiply(x, prod_gate_weights) + prod_gate_bias

        # RESHAPE
        x = tf.reshape(x, [-1, self.n_check_nodes, self.n_variable_nodes])

        # REPEAT
        x = tf.repeat(x, (self.n_variable_nodes - 1), axis=-2)

        # RESHAPE
        x = tf.reshape(
            x,
            [
                -1,
                self.n_check_nodes * self.n_variable_nodes,
                (self.n_variable_nodes - 1),
            ],
        )

        # REDUCE PROD
        x = tf.reduce_prod(x, axis=-1)

        # REVERSE
        x = tf.reverse(x, axis=[-1])

        # 2*ARCTANH
        """
        x = 2 * self.atanh_activation_layer_training(x)
        """
        if training:
            x = 2 * self.atanh_activation_layer_training(x)
        else:
            x = 2 * self.atanh_activation_layer_eval(x)

        new_states = tf.reshape(x, [1, -1, self.n_check_nodes * self.n_variable_nodes])

        ############################ OUTPUT ##########################################
        # Weights
        weighted_new_states_out = tf.multiply(new_states, out_weights)

        # Gate
        gated_new_states_out = tf.multiply(weighted_new_states_out, out_gate)

        # Inputs
        weighted_new_inputs = tf.multiply(
            llr, self.input_weights_out
        )  # weighted_inputs #
        # weighted_new_inputs = weighted_inputs
        x = tf.concat([gated_new_states_out, weighted_new_inputs], axis=-1)

        x = tf.reshape(
            x,
            [
                -1,
                (self.n_check_nodes + 1),
                self.n_variable_nodes,
            ],
        )

        outputs = tf.reduce_sum(x, axis=-2)

        ############################ RETURN ############################

        return tf.reshape(outputs, [-1, 1, self.n_variable_nodes]), tf.reshape(
            new_states, [-1, self.n_check_nodes * self.n_variable_nodes]
        )
