#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Auto-Encoder Structure

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

import warnings


from encoders import Encoder
from channels import AWGN
from decoders import Decoder
from code_generators import CodeGenerator


class AutoEncoder(tf.keras.Model):
    def __init__(
        self,
        n,
        k,
        n_iter,
        conf,
        training_noise_power_db,
        G=None,
        H=None,
        trainable_code=True,
        trainable_decoder=True,
        **kwargs,
    ):
        super(AutoEncoder, self).__init__(**kwargs)
        print(
            f"Create AE model with param: n={n}, k={k}, n_iter={n_iter}, conf={conf}, training_noise_power_db={training_noise_power_db}, G={G}, H={H}, trainable_code={trainable_code}, trainable_decoder={trainable_decoder}"
        )

        self.code_generator = CodeGenerator(
            n, k, G, H, trainable_code, name="code_generator"
        )
        self.encoder = Encoder(n, k, name="encoder")
        self.channel = AWGN(noise_power_db=training_noise_power_db, name="channel")
        self.decoder = Decoder(
            n,
            n - k,
            k,
            n_iter=n_iter,
            trainable=trainable_decoder,
            conf=conf,
            name="decoder",
        )

        if conf == "ML":
            if G == None:
                warnings.warn(
                    "Decoder type was set to ML but no Generator matrix was provided"
                )
            else:  #!!!!!!!!!!!!!! WORKAROUND
                self.decoder.decoder.init_codebook(G)

        if conf in ["BP", "GNBP"]:
            if G == None or H == None:
                warnings.warn(
                    f"Decoder type was set to {conf} but no Generator and/or Parity-Check matrix was provided"
                )

        if not trainable_code:
            if G == None or H == None:
                warnings.warn(
                    f"Code was set to be not trainable but no Generator and/or Parity-Check matrix was provided"
                )

        if G != None:
            self.code_generator.set_G(G)
        if H != None:
            self.code_generator.set_H(H)

        self.n = n
        self.k = k
        self.training_noise_power_db = training_noise_power_db

    def call(self, inputs, training=False):
        (G, H) = self.code_generator(tf.constant([1]), training=training)
        symbols = self.encoder(inputs=[inputs, G], training=training)

        noisy_symbols = self.channel(symbols, training=training)
        sigma2 = self.channel.noise_power
        reconstructed_messages = self.decoder(
            inputs=[noisy_symbols, G, H, sigma2], training=training
        )
        return reconstructed_messages
