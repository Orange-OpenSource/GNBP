#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Additive White Gaussian Noise channel

Brief: Add white gaussian noise to input signal

Copyright (c) 2022 Orange

Author: Quentin Lampin <quentin.lampin@orange.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice (including the next paragraph) shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
"""

import tensorflow as tf

from tools import decibeltolinear, lineartodecibel


class AWGN(tf.keras.layers.Layer):
    def __init__(self, noise_power=None, noise_power_db=None, **kwargs):
        super(AWGN, self).__init__()

        if noise_power is None and noise_power_db is None:
            raise ValueError("provide noise_power or noise_power_db as argument")

        if noise_power_db is not None:
            noise_power = decibeltolinear(noise_power_db)

        self._noise_power = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.Constant(noise_power),
            name="w_noise",
            trainable=False,
        )

    def build(self, _):
        pass

    @property
    def noise_power(self):
        return self._noise_power

    @noise_power.setter
    def noise_power(self, new_noise_power):
        self._noise_power.assign(tf.constant(new_noise_power, shape=(1,)))

    @property
    def noise_power_db(self):
        return lineartodecibel(self._noise_power)

    @noise_power_db.setter
    def noise_power_db(self, new_noise_power_db):
        self._noise_power.assign(
            tf.constant(decibeltolinear(new_noise_power_db), shape=(1,))
        )

    def call(self, inputs, training=False):

        noise_samples = tf.random.normal(
            shape=tf.shape(inputs),
            mean=0.0,
            stddev=tf.sqrt(self._noise_power / 2.0),
        )
        return inputs + noise_samples
