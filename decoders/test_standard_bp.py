#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Brief: validate that the GNBP decoder configured to execute a conventional BP is equivalent to a conventional BP.

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
import numpy as np
import os

from . import Decoder, DecoderStandardBP
from . import SumProduct, FactorGraph


def test_placeholder_decoder_model():
    G = tf.constant(
        [
            [1, 0, 0, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 1, 1, 1],
        ],
        dtype=tf.float32,
    )

    H = tf.constant(
        [
            [1, 1, 1, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 0, 1],
        ],
        dtype=tf.float32,
    )

    n_iter = 5

    llrs = tf.random.uniform(
        shape=[100, tf.shape(H)[1]], minval=-10.0, maxval=+10.0, dtype=tf.float32
    )
    model_a = Decoder(
        n_variable_nodes=tf.shape(H)[1],
        n_check_nodes=tf.shape(H)[0],
        n_information_bits=tf.shape(G)[0],
        n_iter=n_iter,
        trainable=False,
        conf="BP",
    )
    model_b = DecoderStandardBP(
        n_variable_nodes=tf.shape(H)[1],
        n_check_nodes=tf.shape(H)[0],
        n_information_bits=tf.shape(G)[0],
        n_iter=n_iter,
        trainable=False,
    )

    a = model_a([llrs, G, H, 1.0])
    b = model_b([llrs, H, 1.0])
    tf.debugging.assert_equal(a, b)


def test_decoder_model_eq_standard_sum_product():
    #! This test aims to compare the result of a standard BP algorithm and the fensorflow version.
    #! It seems to work only when specifying the datatype to float64 before any atanh or tanh operation
    #! Although it does not necessarily affect the BER performance.
    # Get the code G and H matrices
    n = 31
    k = 16
    codename = f"BCH_{n}_{k}"
    # codename = f"hamming_{n}_{k}"
    code_path = os.path.join(
        "./", "encoders/linearblockencoders_reference/", f"{codename}.npz"
    )

    code_file = np.load(code_path)

    G_sys = tf.convert_to_tensor(code_file["G"], dtype=tf.float32)
    H_sys = tf.convert_to_tensor(code_file["H_systematic"], dtype=tf.float32)
    # H_nsys = tf.convert_to_tensor(code_file['H_non_systematic'], dtype=tf.float32)
    G = G_sys
    H = H_sys
    """
    G = tf.constant(
        [
            [1, 0, 0, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 1, 1, 1],
        ],
        dtype=tf.float32,
    )

    H = tf.constant(
        [
            [1, 1, 1, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 0, 1],
        ],
        dtype=tf.float32,
    )
    """

    systematic_bits = [True] * k + [False] * (n - k)

    for A in [1.0, 10.0, 1e3]:  #![10.0]:  #!
        for n_iter in [1, 5]:  #![5]:  #!
            llrs = tf.random.uniform(
                shape=[100, n], minval=-A, maxval=+A, dtype=tf.float32  #!1000
            )

            model_a = DecoderStandardBP(
                n_variable_nodes=n,  # tf.shape(H)[1],
                n_check_nodes=(n - k),  # tf.shape(H)[0],
                n_information_bits=k,  # tf.shape(G)[0],
                n_iter=n_iter,
                trainable=False,
            )

            model_b = FactorGraph("F1", H, systematic_bits, algorithm=SumProduct())

            sigma = 1.0

            a = model_a([llrs, H, sigma])
            a = tf.constant(tf.math.round(a), dtype=tf.float32)  #!tf.constant(a)  #!

            b = (
                model_b.decode(
                    (-1.0) * 4 * llrs / sigma,
                    max_iteration=n_iter,
                    min_iteration=n_iter,  #!min_iteration=0,#!
                ),
            )
            b = tf.constant(b, dtype=tf.float32)
            """
            tf.debugging.assert_near(
                a,
                b,
                atol=1e-1,
                summarize=-1,
                message=f"x == y did not hold for A={A} and n_iter={n_iter}",
            )
            """
            tf.debugging.assert_equal(
                a,
                b,
                summarize=-1,
                message=f"x == y did not hold for A={A} and n_iter={n_iter}",
            )
