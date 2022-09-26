#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Code Generator Structure

Brief: Define the code generator model structure of the AE model.

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

import tensorflow as tf

from activations import differentiable_step_function, step_function


class CodeGenerator(tf.keras.Model):
    """
    Weight place-holder model used to provide to other module the Generator matrix and associated Parity Check matrix of a SYSTEMATIC linear block code

    Args:
        n (int): code length
        k (int): message length
    """

    def __init__(self, n, k, G=None, H=None, trainable_code=True, **kwargs):
        super(CodeGenerator, self).__init__(**kwargs)
        """
        Place holder model for linear block code containing generator matrix G and parity check matrix H.
        If G and H are not provided the CodeGenerator instantiate a trainable SYSTEMATIC Generator matrix 
        and associated Parity Check matrix. Otherwise the provided matrices are used in place of the trainable 
        ones. If only one of the two matrix is provided it will be used as a non trainable matrix while the 
        other matrix will be kept trainable but in a SYSTEMATIC form.
        Args:
        - n (int): the size of a codeword in bits.
        - k (int): the size of an information word in bits.
        - G (int) [defautl = None]: The [k x n] generator matrix.   #! Dimension?
        - H (int) [defautl = None]: The [(n-k) x n] parity-check matrix.#! Dimension?
        """
        self.n = n
        self.k = k
        self.G = G
        self.H = H
        self.trainable_code = trainable_code

        # self.build(input_shape=(1,))

    def build(self, input_shape):
        self.redundancy_weights_G = self.add_weight(
            shape=(self.k * (self.n - self.k),),  # (self.k*(self.n),),#
            initializer=tf.keras.initializers.RandomUniform(-0.01, +0.01),
            name="redundancy_weights_G",
            trainable=self.trainable_code,
            dtype=tf.float32,
        )

    def call(self, inputs, training=False):
        g = differentiable_step_function(self.redundancy_weights_G)
        h = g  # differentiable_step_function(self.redundancy_weights_G)

        # PC and Generator matrices:
        if self.G == None:
            G = tf.keras.layers.Concatenate()(
                [
                    tf.eye(self.k, dtype=tf.float32),
                    tf.reshape(g, [self.k, (self.n - self.k)]),
                ]
            )
        else:
            G = self.G

        if self.H == None:
            H = tf.keras.layers.Concatenate()(
                [
                    tf.transpose(tf.reshape(h, [self.k, (self.n - self.k)])),
                    tf.eye((self.n - self.k), dtype=tf.float32),
                ]
            )
        else:
            H = self.H

        return (G, H)

    def set_G(self, G):
        """
        Manually set the generator matrix G.
        Once the matrix is set, the generator matrix is not trainable anymore
        except if the matrix is reset to the value of None

        Args:
            G ([2d array] or None): Generator matrix
        """

        if G != None:
            k = tf.shape(G)[0]
            n = tf.shape(G)[1]
            assert (
                k == self.k and n == self.n
            ), (
                -f"The provided generator matrix shape is ({k},{n}). Expected ({self.k},{self.n})"
            )
            self.G = G

        return True

    def set_H(self, H):
        """
        Manually set the parity-check matrix H.
        Once the matrix is set, the parity-check matrix is not trainable anymore
        except if the matrix is reset to the value of None

        Args:
            H ([2d array] or None): parity-check matrix
        """

        if H != None:
            m = tf.shape(H)[0]
            n = tf.shape(H)[1]
            assert (
                m == (self.n - self.k) and n == self.n
            ), (
                -f"The provided parity-check matrix shape is ({m},{n}). Expected ({(self.n-self.k)},{self.n})"
            )
            self.H = H

        return True
