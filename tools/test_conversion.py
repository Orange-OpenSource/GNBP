#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from . import decibeltolinear, lineartodecibel, ebno_db_to_snr_db, ebno_to_snr


def test_lineartodecibel():
    linear = tf.constant(2, dtype=tf.float32)
    decibel = lineartodecibel(linear)
    assert decibel == 3.0103


def test_decibeltolinear():
    decibel = tf.constant(3.0, dtype=tf.float32)
    linear = decibeltolinear(decibel)
    tf.debugging.assert_near(linear, 1.9952623)


def test_ebno_db_to_snr_db():
    ebn0_db = tf.constant(3.0, dtype=tf.float32)
    bitspersymbol = tf.constant(2, dtype=tf.float32)
    snr_db = ebno_db_to_snr_db(ebn0_db, bitspersymbol)
    assert snr_db == 6.0102997


def test_ebno_to_snr():
    """convert Eb/N0 to SNR

    Args:
        x (tf.Tensor): Eb/No
        bitspersymbols (tf.float32): bits per symbol

    Returns:
        tf.Tensor: SNR values
    """
    ebn0 = tf.constant(0.5, dtype=tf.float32)
    bitspersymbol = tf.constant(2, dtype=tf.float32)
    snr = ebno_to_snr(ebn0, bitspersymbol)
    assert snr == 1


if __name__ == "__main__":
    pytest.main()
