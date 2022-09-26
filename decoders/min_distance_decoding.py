"""
ML Decoder

Brief: Basic ML decoder using exhaustiv minimal distance calculation.

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
from tensorflow.python.ops.gen_array_ops import expand_dims

#!!!! WARNING: CODE BOOK CREATION METHODS TO BE THOROUGHLY CHECKED: POTENTIAL ROUNDING/TYPE/OVERFLOW ERRORS
class MinDistanceDecoder(tf.keras.Model):
    def __init__(self, n, k, G=None, return_words=True, **kwargs):
        super(MinDistanceDecoder, self).__init__(**kwargs)
        self.n = n
        self.k = k
        self.number_of_words = int(tf.pow(2, self.k))
        self.G = G
        self.return_words = return_words
        self.code_book_generated = False

        if G != None:
            self.init_codebook(G)

    def init_codebook(self, G):
        n = tf.shape(G)[1]
        k = tf.shape(G)[0]
        assert (
            k == self.k and n == self.n
        ), (
            -f"The provided generator matrix shape ({k},{n}) doesn't match expected shape ({self.k},{self.n})"
        )

        self.G = tf.cast(G, dtype=tf.float32)

        # Compute all possible input words
        coefs = tf.math.round(
            tf.pow(2.0 * tf.ones(self.k), tf.linspace(1.0, float(self.k), self.k))
        )
        self.possible_words = (
            tf.floor(
                tf.linspace(
                    tf.zeros(self.k),
                    float(self.number_of_words) - 1,
                    self.number_of_words,
                    axis=0,
                )
            )
            % (coefs)
        )
        substraction_slice = tf.concat(
            [tf.zeros([self.number_of_words, 1]), self.possible_words[:, :-1]],
            axis=-1,
        )
        self.possible_words = tf.cast(
            (self.possible_words - substraction_slice) / (coefs / 2), dtype=tf.float32
        )

        # tf.print(self.possible_words, summarize=-1)
        # Compute all possible code words
        self.possible_codewords = tf.matmul(self.possible_words, self.G) % 2

        # optimal LLR - Theoretically should be -infinity and +infinity
        self.possible_codewords_llrs = (-1) * (
            self.possible_codewords * 2 - 1
        )  # * (-100000)

        self.code_book_generated = True
        return self.code_book_generated

    def call(self, inputs, training=False):
        noisy_symbols, G, sigma2 = inputs
        noisy_symbols = tf.cast(noisy_symbols, dtype=tf.float32)
        llrs = (-1.0) * 4.0 * noisy_symbols / sigma2

        if not self.code_book_generated:
            self.init_codebook(G)

        # Expand
        x = tf.expand_dims(llrs, axis=-2)

        # Broadcast
        batch_size = tf.shape(x)[0]
        x = tf.broadcast_to(x, shape=[batch_size, self.number_of_words, self.n])
        # tf.print(x, summarize=-1)
        # tf.print(self.possible_codewords, summarize=-1)

        # Sum abs difference
        max_abs_x = tf.reduce_max(tf.abs(x))
        # tf.print(max_abs_x)
        x = tf.reduce_mean(
            tf.abs(x - max_abs_x * self.possible_codewords_llrs), axis=-1
        )
        # tf.print(x, summarize=-1)
        # Argmin
        x = tf.argmin(x, axis=-1)
        # tf.print(x, summarize=-1)

        if self.return_words:
            # Decoded words
            output = tf.gather(self.possible_words, x)
        else:
            # Decoded codewords
            output = tf.gather(self.possible_codewords, x)
        # tf.print(output, summarize=-1)
        return output

    """
    def decode(
        self, received_code_words, return_words=True
    ):  # received code word in the forms of LLR!
        self.return_words = return_words
        return self.call(received_code_words)
    """
