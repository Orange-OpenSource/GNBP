#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""BPSK Modulation Layer

Tensorflow layer defining a BPSK modulation

Brief: BPSK Modulation Layer

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
from activations import differentiable_sign_function


class DifferentiableBPSKModulationLayer(tf.keras.layers.Layer):
    def __init__(self, binary_inputs=True, differentiable_approximation=True, **kwargs):
        super(DifferentiableBPSKModulationLayer, self).__init__(**kwargs)
        """
        generate BPSK symbols using a hard sign function (forward pass) with differentiable approximation (backward pass).
        Info can either be binary encoded [0 ; 1] or encoded in the sign of the input [0 <-> + ; 1 <-> - ]
        The output symbols are encoded as follows: [0 -> -1 ; 1 -> +1]
        Arg:
            binary_inputs (bool) [default=True]: wheter the inputs are provided as binary input [0,1] or logits [-,+]
            differnetiable_approximation (bool) [default=True]: wheter a differentiable approximation should be used rather as a true step function to allow for gradient back propagation
        """
        self.binary_inputs = binary_inputs
        self.differentiable_approximation = differentiable_approximation

    def call(self, inputs, training=False):
        # tf.print("BPSK",inputs,summarize=-1)
        if self.binary_inputs:
            inputs = inputs - 1 / 2  # Binary inputs
        else:
            inputs = -inputs  # Sign inputs

        if self.differentiable_approximation:
            outputs = differentiable_sign_function(inputs)
        else:
            outputs = tf.where(tf.greater(inputs, 0), 1.0, -1.0)
        # tf.print("BPSK",outputs,summarize=-1)
        return outputs
