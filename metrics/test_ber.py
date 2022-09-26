"""
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

import pytest

import tensorflow as tf

from . import BitErrorRate


def test_ber_logits():
    # errors in positions 1, 3 and high uncertainty on position 6
    bits_truth = tf.constant([[0, 0, 1, 1, 1, 1, 0, 1]], dtype=tf.float32)
    logits = tf.constant([[-10, 3, 5, -2, 1, 23, 0, 2]], dtype=tf.float32)

    expected_ber = 0.25

    ber_metric = BitErrorRate(from_logits=True)
    ber = ber_metric(y_true=bits_truth, y_pred=logits)
    assert ber == expected_ber


def test_ber_nologits():
    # errors in positions 5, 6, 7
    bits_truth = tf.constant([[0, 0, 1, 1, 1, 1, 0, 1]], dtype=tf.float32)
    bits_pred = tf.constant([[0, 0, 1, 1, 1, 0, 1, 0]], dtype=tf.float32)

    expected_ber = 3.0 / 8

    ber_metric = BitErrorRate(from_logits=False)
    ber = ber_metric(y_true=bits_truth, y_pred=bits_pred)
    assert ber == expected_ber


def test_ber_average_across_batches():
    # errors in positions 5, 6, 7
    bits_truth = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
    bits_pred = tf.constant([[0, 0, 1, 1, 1, 0, 1, 0]], dtype=tf.float32)

    expected_ber = 4.0 / 16.0
    ber_metric = BitErrorRate(from_logits=False)
    ber = ber_metric(y_true=bits_truth, y_pred=bits_pred)
    ber = ber_metric(y_true=bits_truth, y_pred=bits_truth)
    assert ber == expected_ber


if __name__ == "__main__":
    pytest.main()
