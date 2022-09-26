#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Binomial proportion confidence interval based early stopping callback for evaluation

This callback monitors a binomial distributed loss or metric, evaluates the confidence interval
and interrupt the evaluation when the confidence interval span to mean value is below a given 
ratio.

Brief: binomial proportion confidence interval based early stopping callback for evaluation

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


class BatchTerminationCallback(tf.keras.callbacks.Callback):
    def __init__(self, condition):
        super(BatchTerminationCallback, self).__init__()
        self.condition = condition

    def on_test_batch_end(self, batch, logs=None):
        condition = self.condition(batch, logs)
        if condition:
            print("stopping evaluation")
            raise StopIteration("termination.")
