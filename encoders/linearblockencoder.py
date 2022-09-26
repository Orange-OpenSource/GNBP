#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Linear block encoder

Encode k-sized blocks of digits into n-sized blocks of digits using a linear transformation
described by the generator matrix G of size (k, n)

Brief: Linear block encoder

Copyright (c) 2022 Orange

Authors: Guillaume Larue <guillaume.larue@orange.com>, Quentin Lampin <quentin.lampin@orange.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice (including the next paragraph) shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
"""

import tensorflow as tf


class ProductWithExternalWeights(tf.keras.layers.Layer):
    """
    GF2 liner transformation where the modulo 2 operation (XOR) is replaced by using a bipolar product
    e.g. 0 XOR 1 XOR 1 = 0 --> +1 x -1 x -1 = +1

    Args:
        ###FALSE IN CURRENT VERSION. units (int): size of the layer output (the dimension of the wieght matrix is dependant on the input size which can be defined dynamically as in a Dense layer)
        activation (tf.keras.layers.Activation) [default: None]: The activation function to apply to the output of the layer
    """

    def __init__(
        self,
        units,
        activation=None,
        **kwargs,
    ):
        super(ProductWithExternalWeights, self).__init__(**kwargs)
        # self.input_spec = InputSpec(ndim=2)
        self.units = units
        self.activation = activation

    def build(self, input_shape):

        if self.activation is not None:
            self.activation_layer = tf.keras.layers.Activation(self.activation)

    @tf.function
    def f(self, w):
        f = w
        return f

    @tf.function
    def g(self, w):
        g = 1 - w
        return g

    def call(self, inputs):
        (inputs, weights) = inputs
        self.kernel = tf.reshape(weights, [tf.shape(inputs)[-1], self.units])
        repeated_inputs = tf.keras.backend.repeat(inputs, n=1)
        x = tf.multiply(repeated_inputs, self.f(tf.transpose(self.kernel))) + self.g(
            tf.transpose(self.kernel)
        )

        if self.activation is not None:
            output = self.activation_layer(tf.reduce_prod(x, axis=-1))
        else:
            output = tf.reduce_prod(x, axis=-1)

        return output


class LinearBlockCodeProductEncoderWithExternalG(tf.keras.layers.Layer):
    """Linear block encoder

    Encode k-sized blocks of digits into n-sized blocks of digits using a linear transformation
    described by the generator matrix G of size (k, n). The GF2 modulo operation is replaced for differentiability purpose by the equivalent form based on bipolar products.
    The G matrix weights are provided by an external placeholder model.

    Args:
        n (int): code length
        k (int): message length
        return_binary (bool) [default: True]: whether the model should apply a sigmoid activation to return binary like outputs
    """

    def __init__(
        self,
        n,
        k,
        return_binary=True,
        **kwargs,
    ):
        super(LinearBlockCodeProductEncoderWithExternalG, self).__init__(**kwargs)
        # Code properties
        self.n = n
        self.k = k
        # Encoder properties
        self.return_binary = return_binary
        # Layers definition
        self.product = ProductWithExternalWeights(units=self.n)
        ## Build model
        # self.build_graph(input_shape=(1, self.k))

    def call(self, inputs, training=False):
        (inputs, G) = inputs
        x = 1 - inputs * 2
        x = self.product(inputs=[x, G])
        if self.return_binary:
            x = tf.sigmoid(-x)
        return x

    """
    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)

        if not hasattr(self, "call"):
            raise AttributeError("User should define 'call' method in sub-class model!")

        _ = self.call(inputs)
    """
