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

from . import BlockErrorRate


def test_bler_logits():
    # 1 bit error in block 3
    block_truth = tf.constant(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=tf.float32,
    )
    logits = tf.constant(
        [
            [-10, -10, -10, -10, -10, -10, -10, -10],
            [-10, -10, -10, -10, -10, -10, -10, -10],
            [-10, -10, -10, -10, -10, -10, -10, +10],
            [-10, -10, -10, -10, -10, -10, -10, -10],
        ],
        dtype=tf.float32,
    )

    expected_bler = 0.25

    bler_metric = BlockErrorRate(from_logits=True)
    bler = bler_metric(y_true=block_truth, y_pred=logits)
    assert bler == expected_bler


def test_bler_nologits():
    # errors in positions 5, 6, 7
    block_truth = tf.constant(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=tf.float32,
    )
    block_pred = tf.constant(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=tf.float32,
    )

    expected_bler = 0.25

    bler_metric = BlockErrorRate(from_logits=False)
    bler = bler_metric(y_true=block_truth, y_pred=block_pred)
    assert bler == expected_bler


if __name__ == "__main__":
    pytest.main()
