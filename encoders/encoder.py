#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Encoder

Tensorflow based encoder model for linear block codes

Brief: Tensorflow based encoder model for linear block codes

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
from encoders import LinearBlockCodeProductEncoderWithExternalG
from encoders import DifferentiableBPSKModulationLayer


class Encoder(tf.keras.Model):
    def __init__(self, n, k, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.n = n
        self.k = k
        self.lbcpe = LinearBlockCodeProductEncoderWithExternalG(
            self.n, self.k, return_binary=True
        )
        self.modulation = DifferentiableBPSKModulationLayer(
            binary_inputs=True, differentiable_approximation=True
        )

    def call(self, inputs, training=False):
        (inputs, G) = inputs
        x = self.lbcpe(inputs=[inputs, G], training=training)
        output_symbols = self.modulation(x, training=training)
        return output_symbols
