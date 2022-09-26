#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bit error count metric

A keras metric to evaluate the BEC (Bit Error Count) of a system

Brief: BEC (Bit Error Count) metric

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


class BitErrorCount(tf.keras.metrics.Metric):
    def __init__(self, name="BEC", from_logits=False, mode=None, **kwargs):
        """Bit Error Rate metric

        Args:
            name (str, optional): metric's name. Defaults to 'BEC'.
            from_logits (bool, optional): evaluate the BER from logits? Defaults to False.
            mode (str, optional): Defaults to None. 'average': average accross batches, 'sum': sum over batches.
        """
        super(BitErrorCount, self).__init__(name=name, **kwargs)
        self.bec = self.add_weight(name="BEC", initializer="zeros", dtype=tf.float32)
        self.n = self.add_weight(name="n", initializer="zeros", dtype=tf.float32)
        self.from_logits = from_logits
        self.mode = mode

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits == True:
            y_pred = tf.math.sign(y_pred)
            y_pred += 1
            y_pred /= 2
        y_pred = tf.clip_by_value(y_pred, 0, 1)
        y_pred = tf.round(y_pred)
        differences = tf.abs(y_true - y_pred)
        bec = tf.reduce_sum(differences)
        if self.mode == "average":
            count = tf.cast(tf.size(y_pred), dtype=tf.float32)
            bec = self.bec * self.n + count * bec
            self.n.assign_add(count)
            self.bec.assign(bec / self.n)
        elif self.mode == "sum":
            self.bec.assign_add(bec)
        else:
            self.bec.assign(bec)

    def result(self):
        return self.bec

    def reset_state(self):
        self.bec.assign(0.0)
        self.n.assign(0.0)
        tf.print(f"{self.name} is reset")

    def get_config(self):
        return {"from_logits": self.from_logits}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
