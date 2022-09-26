#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Decibel to linear and linear to decibel conversion

Utility functions for converting from the decibel scale to the linear and vice-versa

Brief: Decibel to linear and linear to decibel conversion functions

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


@tf.function
def lineartodecibel(x):
    """convert values to decibel scale

    Args:
        x (tf.tensor): values to convert

    Returns:
        tf.tensor: converted values
    """
    return 10.0 * tf.math.log(x) / tf.math.log(10.0)


@tf.function
def decibeltolinear(x):
    """convert values expressed in decibel scale into linear scale

    Args:
        x (tf.tensor): values to convert

    Returns:
        tf.tensor: converted values
    """
    return tf.exp(x * tf.math.log(10.0) / 10.0)


def ebno_db_to_snr_db(x, bitspersymbols):
    """convert Eb/N0 in dB to SNR in dB

    Args:
        x (tf.Tensor): Eb/N0 values in dB
        bitspersymbols (tf.float32): bits per symbol

    Returns:
        tf.Tensor: SNR values in dB
    """
    return x + lineartodecibel(bitspersymbols)


def ebno_to_snr(x, bitspersymbols):
    """convert Eb/N0 to SNR

    Args:
        x (tf.Tensor): Eb/No
        bitspersymbols (tf.float32): bits per symbol

    Returns:
        tf.Tensor: SNR values
    """
    return x * bitspersymbols
