#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Regularizers functions for model weights

Brief: Definitions of regularizers functions for model weights

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


class L1WeightRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, alpha=1, mean=1):
        self.alpha = alpha
        self.mean = mean

    def __call__(self, x):
        return self.alpha * tf.reduce_mean(tf.abs((x - self.mean)))

    def get_config(self):
        return {"alpha": float(self.alpha), "mean": float(self.mean)}


class L2WeightRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, alpha=1, mean=1):
        self.alpha = alpha
        self.mean = mean

    def __call__(self, x):
        return self.alpha * tf.reduce_mean(tf.square((x - self.mean)))

    def get_config(self):
        return {"alpha": float(self.alpha), "mean": float(self.mean)}
