"""
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

import pytest
import tensorflow as tf

from .linearblockencoder import (
    LinearBlockCodeProductEncoderWithExternalG,
)


def test_linearcoder_with_external_G_product():
    x = tf.constant(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=tf.float32,
    )
    y_true = tf.constant(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 1],
            [1, 0, 0, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 0, 1, 0],
            [1, 1, 0, 0, 0, 1, 1],
            [1, 1, 0, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=tf.float32,
    )
    hamming74_g = tf.constant(
        [
            [1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=tf.float32,
    )
    hamming74 = LinearBlockCodeProductEncoderWithExternalG(n=7, k=4, return_binary=True)

    y_pred = tf.math.round(hamming74([x, hamming74_g]))

    tf.debugging.assert_equal(y_pred, y_true)


if __name__ == "__main__":
    pytest.main()
