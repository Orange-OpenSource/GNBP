#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Activation Functions

Brief: Exposes various custom activation functions related to BP algorithm.

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

from tensorflow.keras.layers import Layer
from tensorflow import clip_by_value, atanh
import tensorflow as tf


class AtanhActivation(Layer):
    def __init__(self, clip_value=+20.0, **kwargs):
        super(AtanhActivation, self).__init__(**kwargs)
        self.clip_value = clip_value

    def call(self, inputs):
        return clip_by_value(
            atanh(inputs),
            clip_value_min=-self.clip_value,
            clip_value_max=+self.clip_value,
        )
        # return clipped_grad_atanh_function(inputs)


class AtanhTaylorApproxActivation(Layer):
    def __init__(self, order=21, **kwargs):
        super(AtanhTaylorApproxActivation, self).__init__(**kwargs)
        self.order = order
        self.coefficients = tf.range(1, order + 1, 2, dtype=tf.float32)
        # tf.print(self.coefficients,summarize=-1)

    def call(self, inputs):
        # tf.print(inputs,summarize=-1)
        x = tf.expand_dims(inputs, axis=-1)
        # tf.print(x,summarize=-1)
        x = tf.repeat(x, int((self.order - 1) / 2 + 1), axis=-1)
        # tf.print(x,summarize=-1)
        x = (x ** self.coefficients) / self.coefficients
        # tf.print(x,summarize=-1)
        x = tf.reduce_sum(x, axis=-1)
        return x


@tf.custom_gradient
def differentiable_sign_function(x):
    # result = tf.sign(x) # forward computation
    result = tf.where(tf.greater(x, 0), 1.0, -1.0)
    # result = tf.where(tf.greater(x,0),0.99,-0.99) #Non binary output improve trainability?
    def custom_grad(dy):
        alpha = 1
        grad = dy * (
            1 - tf.square(tf.tanh(alpha * x))
        )  # backward computation -> compute gradient (Tanh gradient)
        # grad = dy * (1/tf.square(1+tf.abs(alpha*x))) # backward computation -> compute gradient (softsign gradient)
        return grad

    return result, custom_grad


@tf.custom_gradient
def step_function(x):
    result = tf.where(tf.greater(x, 0), 1.0, 0.0)

    def custom_grad(dy):
        grad = dy * 0
        return grad

    return result, custom_grad


@tf.custom_gradient
def differentiable_step_function(x):
    # forward computation
    result = tf.where(tf.greater(x, 0), 1.0, 0.0)

    def custom_grad(dy):
        alpha = 1
        # grad = dy * (1-tf.square(tf.tanh(alpha*x))) # backward computation -> compute gradient (Tanh gradient)
        grad = (
            alpha * dy * tf.sigmoid(x) * (1 - tf.sigmoid(x))
        )  # backward computation -> compute gradient (sigmoid gradient)
        # grad = alpha*dy*(tf.where(tf.greater(x,0),+0.1,+0.1)+tf.sigmoid(x)*(1-tf.sigmoid(x))) # backward computation -> compute gradient ("leaky" sigmoid gradient)
        # grad = dy * ((tf.sign(x)+1)/2) # backward computation -> compute gradient (relu gradient)
        # grad = alpha * dy * 1 # backward computation -> compute gradient of identity function(f(x)=x gradient)
        # grad = dy
        return grad

    return result, custom_grad
