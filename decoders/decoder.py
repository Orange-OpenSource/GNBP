"""
Linear Block Codes Decoder Structures

Brief: Exposes various decoder structures BP, GNBP, ML, etc. used in this work. Conf A correspond to the GNBP decoder used in this work.

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

from decoders import GatedNeuralBeliefPropagationRNNCell, MinDistanceDecoder


class Decoder(tf.keras.Model):
    def __init__(
        self,
        n_variable_nodes,
        n_check_nodes,
        n_information_bits,
        n_iter=5,
        trainable=True,
        conf="A",
        **kwargs,
    ):
        super(Decoder, self).__init__(**kwargs)
        self.n_variable_nodes = n_variable_nodes
        self.n_check_nodes = n_check_nodes
        self.n_information_bits = n_information_bits
        self.n_iter = n_iter
        self.trainable = trainable
        self.conf = conf

        print("CONF:", conf)
        if conf == "A":
            self.decoder = DecoderA(
                n_variable_nodes, n_check_nodes, n_information_bits, n_iter, trainable
            )
        elif conf == "ML":
            self.decoder = MinDistanceDecoder(
                n=n_variable_nodes, k=n_information_bits, G=None, return_words=True
            )

        elif conf == "BP":
            self.decoder = DecoderStandardBP(
                n_variable_nodes, n_check_nodes, n_information_bits, n_iter, trainable
            )

        elif conf == "GNBP":
            self.decoder = self.decoder = DecoderA(
                n_variable_nodes, n_check_nodes, n_information_bits, n_iter, trainable
            )

        else:
            print("default configuration")
            self.decoder = DecoderA(
                n_variable_nodes, n_check_nodes, n_information_bits, n_iter, trainable
            )

    def call(self, inputs, training=False):
        (noisy_symbols, G, H, sigma2) = inputs

        if self.conf == "ML":
            return self.decoder([noisy_symbols, G, sigma2], training=training)
        else:
            return self.decoder([noisy_symbols, H, sigma2], training=training)


class DecoderA(tf.keras.Model):
    def __init__(
        self,
        n_variable_nodes,
        n_check_nodes,
        n_information_bits,
        n_iter=5,
        trainable=True,
        **kwargs,
    ):
        super(DecoderA, self).__init__(**kwargs)
        self.n_variable_nodes = n_variable_nodes
        self.n_check_nodes = n_check_nodes
        self.n_information_bits = n_information_bits
        self.n_iter = n_iter
        self.trainable = trainable

        self.RNN_cell = GatedNeuralBeliefPropagationRNNCell(  #! Atanh taylor during training and true Atanh during eval
            n_variable_nodes=self.n_variable_nodes,
            n_check_nodes=self.n_check_nodes,
            trainable=self.trainable,
        )
        self.SP_RNN = tf.keras.layers.RNN(
            self.RNN_cell,
            return_sequences=True,
            trainable=self.trainable,
        )

        ## Build model
        # self.build_graph(input_shape=(1, self.n_variable_nodes))

    def build(self, input_shape):
        self.input_ponderation = self.add_weight(
            shape=(self.n_iter, 1),
            initializer=tf.keras.initializers.Constant(1.0),
            name="input_ponderation",
            trainable=self.trainable,
        )

        n_out = self.n_iter
        self.out_ponderation = self.add_weight(
            shape=(n_out, 1),
            initializer=tf.keras.initializers.Constant(1.0),
            name="out_ponderation",
            trainable=self.trainable,
        )

        self.skip_connection_ponderation = self.add_weight(
            shape=(1, 1),
            initializer=tf.keras.initializers.Constant(1.0),
            name="skip_connection_ponderation",
            trainable=self.trainable,
        )

    """
    def build_graph(self, input_shape):
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        _ = self.call(inputs)
    """

    def call(self, inputs, training=False):
        (inputs, H, sigma2) = inputs
        # LLRs
        llrs = (-1.0) * 4.0 * inputs / sigma2

        # Input normalization ("codeword-wise"):
        if self.trainable:
            reduce_mean = tf.reduce_mean(tf.abs(llrs), axis=-1)
            normalized_llrs = llrs / tf.reshape(
                reduce_mean, [tf.shape(reduce_mean)[0], 1]
            )
        else:
            normalized_llrs = llrs

        # Input broadcasting:
        x = tf.expand_dims(normalized_llrs, axis=1)
        x = tf.tile(x, [1, self.n_iter, 1])

        # Input ponderation
        x = tf.multiply(x, self.input_ponderation)

        # Decoding
        decoded_bits = self.SP_RNN(inputs=x, constants=H, training=training)

        outputs = tf.reshape(decoded_bits, (-1, self.n_iter, self.n_variable_nodes))
        # Remove added broadcast dimension
        # Normalization (because skip/residual connections?)
        if self.trainable:
            reduce_mean_out = tf.reduce_mean(tf.abs(outputs), axis=-1)
            normalized_outputs = outputs / tf.reshape(
                reduce_mean_out,
                [tf.shape(reduce_mean_out)[0], tf.shape(reduce_mean_out)[1], 1],
            )
        else:
            normalized_outputs = outputs

        outputs = tf.reduce_mean(
            tf.multiply(self.out_ponderation, normalized_outputs), axis=-2
        )

        # Skip connection
        skip_connection = tf.multiply(self.skip_connection_ponderation, normalized_llrs)
        outputs = tf.add(outputs, skip_connection)

        outputs = tf.math.sigmoid((-1.0) * outputs[:, 0 : self.n_information_bits])

        return outputs


class DecoderStandardBP(tf.keras.Model):
    def __init__(
        self,
        n_variable_nodes,
        n_check_nodes,
        n_information_bits,
        n_iter=5,
        trainable=False,
        **kwargs,
    ):
        super(DecoderStandardBP, self).__init__(**kwargs)
        self.n_variable_nodes = n_variable_nodes
        self.n_check_nodes = n_check_nodes
        self.n_information_bits = n_information_bits
        self.n_iter = n_iter
        self.trainable = trainable

        self.RNN_cell = GatedNeuralBeliefPropagationRNNCell(  #! Atanh taylor during training and true Atanh during eval
            n_variable_nodes=self.n_variable_nodes,
            n_check_nodes=self.n_check_nodes,
            trainable=False,
        )
        self.SP_RNN = tf.keras.layers.RNN(
            self.RNN_cell,
            return_sequences=False,
            trainable=False,
        )

        ## Build model
        # self.build_graph(input_shape=(1, self.n_variable_nodes))

    # def build(self, input_shape):

    def call(self, inputs, training=False):
        (inputs, H, sigma2) = inputs

        # LLRs
        llrs = (-1.0) * 4 * inputs / sigma2

        # Input broadcasting:
        x = tf.expand_dims(llrs, axis=1)
        x = tf.tile(x, [1, self.n_iter, 1])

        # Decoding
        decoded_bits = self.SP_RNN(inputs=x, constants=H, training=training)

        outputs = tf.reshape(decoded_bits, (-1, self.n_variable_nodes))

        outputs = tf.math.sigmoid((-1.0) * outputs[:, 0 : self.n_information_bits])

        return outputs
