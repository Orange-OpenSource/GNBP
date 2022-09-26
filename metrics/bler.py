#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Frame error Rate metric

a keras metric to evaluate the FER (Frame Error Rate) of a system

Brief: FER (Frame Error Rate) metric

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


class BlockErrorRate(tf.keras.metrics.Metric):
    def __init__(self, name="BLER", from_logits=False, **kwargs):
        """Bit Error Rate metric

        Args:
            name (str, optional): metric's name. Defaults to 'BLER'.
            from_logits (bool, optional): evaluate the BER from logits? Defaults to False.
        """
        super(BlockErrorRate, self).__init__(name=name, **kwargs)
        self.from_logits = from_logits
        self.errors = self.add_weight(
            name="errors", initializer="zeros", dtype=tf.float32
        )
        self.total = self.add_weight(
            name="total", initializer="zeros", dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits == True:
            y_pred = tf.math.sign(y_pred)
            y_pred += 1
            y_pred /= 2
        y_pred = tf.clip_by_value(y_pred, 0, 1)
        y_pred = tf.round(y_pred)

        bit_errors = tf.abs(y_true - y_pred)
        block_errors = tf.reduce_max(bit_errors, axis=-1)
        errors_count = tf.reduce_sum(block_errors)
        self.errors.assign_add(errors_count)

        blocks_count = tf.cast(tf.size(block_errors), dtype=tf.float32)
        self.total.assign_add(blocks_count)

    def reset_state(self):
        self.errors.assign(0.0)
        self.total.assign(0.0)
        # tf.print(f'{self.name} is reset')

    def result(self):
        return self.errors / self.total

    def get_config(self):
        return {"from_logits": self.from_logits}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
