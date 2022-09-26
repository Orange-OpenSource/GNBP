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

from . import BitErrorCount


def test_bec_logits():
    # errors in positions 1, 3 and high uncertainty on position 6
    bits_truth = tf.constant([[0, 0, 1, 1, 1, 1, 0, 1]], dtype=tf.float32)
    logits = tf.constant([[-10, 3, 5, -2, 1, 23, 0, 2]], dtype=tf.float32)

    expected_bec = 2.0

    bec_metric = BitErrorCount(from_logits=True)
    bec = bec_metric(y_true=bits_truth, y_pred=logits)
    assert bec == expected_bec


def test_bec_nologits():
    # errors in positions 5, 6, 7
    bits_truth = tf.constant([[0, 0, 1, 1, 1, 1, 0, 1]], dtype=tf.float32)
    bits_pred = tf.constant([[0, 0, 1, 1, 1, 0, 1, 0]], dtype=tf.float32)

    expected_bec = 3.0

    bec_metric = BitErrorCount(from_logits=False)
    bec = bec_metric(y_true=bits_truth, y_pred=bits_pred)
    assert bec == expected_bec


def test_bec_average():
    # errors in positions 5, 6, 7
    bits_truth = tf.constant([[0, 0, 1, 1, 1, 1, 0, 1]], dtype=tf.float32)
    bits_pred = tf.constant([[0, 0, 1, 1, 1, 0, 1, 0]], dtype=tf.float32)

    expected_bec = 3.0

    bec_metric = BitErrorCount(from_logits=False, mode="average")
    bec = bec_metric(y_true=bits_truth, y_pred=bits_pred)
    bec = bec_metric(y_true=bits_truth, y_pred=bits_pred)
    assert bec == expected_bec


def test_bec_sum():
    # errors in positions 5, 6, 7
    bits_truth = tf.constant([[0, 0, 1, 1, 1, 1, 0, 1]], dtype=tf.float32)
    bits_pred = tf.constant([[0, 0, 1, 1, 1, 0, 1, 0]], dtype=tf.float32)

    expected_bec = 6.0

    bec_metric = BitErrorCount(from_logits=False, mode="sum")
    bec = bec_metric(y_true=bits_truth, y_pred=bits_pred)
    bec = bec_metric(y_true=bits_truth, y_pred=bits_pred)
    assert bec == expected_bec


if __name__ == "__main__":
    pytest.main()
