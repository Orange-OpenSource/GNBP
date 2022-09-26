#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Binomial proportion confidence interval

This metric monitors a binomial distributed loss or metric and evaluates the confidence interval.

Brief: binomial proportion confidence interval

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


class BinomialProportionConfidenceInterval(tf.keras.metrics.Metric):
    def __init__(
        self,
        monitor_class,
        monitor_params=None,
        fraction=0.95,
        dimensions=None,
        name="BPCI",
        **kwargs
    ):
        """Metric that monitors a binomial distributed loss or metric and evaluates the confidence interval.

        Args:
            monitored (tf.keras.metrics.Metric): metric to be monitored
            fraction (float, optional): fraction of the values in the interval.
            dimension (list|integer, optional): dimensions to consider for counting. Defaults to 'None'.
            name (str, optional): name of the metric. Defaults to 'BPCI'.
        """
        super(BinomialProportionConfidenceInterval, self).__init__(name=name, **kwargs)
        self.monitored_metric = monitor_class(**monitor_params)
        self.fraction = fraction
        self.dimensions = dimensions
        self.alpha = 1 - fraction
        self.z = 1.0 / (1 - self.alpha / 2)

        self.n = self.add_weight(name="n", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.monitored_metric.update_state(y_true, y_pred, sample_weight)
        if self.dimensions is None:
            batch_count = tf.cast(tf.size(y_pred), dtype=tf.float32)
        else:
            shape = tf.cast(tf.shape(y_pred), dtype=tf.float32)
            batch_count = tf.reduce_prod([shape[d] for d in self.dimensions])
        self.n.assign_add(batch_count)

    def reset_state(self):
        self.monitored_metric.reset_state()
        self.n.assign(0.0)

    def result(self):
        value = self.monitored_metric.result()
        n_tilde = self.n + self.z ** 2
        k = value * self.n

        if k == 0.0:
            half_span = 3.0 / (2 * self.n)
            confidence_interval = (0.0, 3.0 / self.n)
        else:
            p_tilde = (1.0 / n_tilde) * (k + (self.z ** 2) / 2)
            half_span = self.z * tf.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
            confidence_interval = (p_tilde - half_span, p_tilde + half_span)
        return (2 * half_span, confidence_interval[0], value, confidence_interval[1])

    def get_config(self):
        return {"monitored_metric": self.monitored_metric, "n": self.n}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
