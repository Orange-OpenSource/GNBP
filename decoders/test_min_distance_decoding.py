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
from . import MinDistanceDecoder


def test_codebook():
    G = tf.constant([[1, 0, 1], [0, 1, 1]], dtype=tf.float32)
    model = MinDistanceDecoder(n=3, k=2, G=G, return_words=True)
    true_codebook = tf.constant(
        [[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=tf.float32
    )
    tf.debugging.assert_equal(
        model.possible_codewords,
        true_codebook,
    )


def test_min_distance_decoding():
    G = tf.constant([[1, 0, 1], [0, 1, 1]], dtype=tf.float32)
    model = MinDistanceDecoder(n=3, k=2, G=G, return_words=True)
    noisy_symbols = tf.constant(
        [
            [-1, -1, -1],
            [-1, +1, +1],
            [+1, -1, +1],
            [+1, +1, -1],
            [-1, -1, +0.5],  # 1 Error
            [-1, -0.5, +1],  # 1 Error
            [-0.5, -1, +1],  # 1 Error
            [+1, -0.5, -1],  # 1 Error
        ],
        dtype=tf.float32,
    )
    sigma2 = tf.constant(1.0, dtype=tf.float32)
    true_ML_decoding_outputs = tf.constant(
        [[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1]],
        dtype=tf.float32,
    )
    tf.debugging.assert_equal(
        model([noisy_symbols, G, sigma2]), true_ML_decoding_outputs
    )
