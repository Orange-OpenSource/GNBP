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
from tools import lineartodecibel

from . import SimpleAWGN


def test_awgn_linear_argument():
    noise_power = tf.constant(1e-4, dtype=tf.float32)
    channel = SimpleAWGN(noise_power=noise_power)
    zero_signal = tf.zeros(shape=(1_000_000, 2))
    noisy_signal = channel(zero_signal)
    noise = noisy_signal
    noise_power_channel = tf.reduce_mean(tf.reduce_sum(tf.square(noise), axis=-1))
    tf.debugging.assert_near(noise_power_channel, noise_power, atol=1e-5)


def test_awgn_decibel_argument():
    noise_power_db = tf.constant(-3.0, dtype=tf.float32)
    channel = SimpleAWGN(noise_power_db=noise_power_db)
    zero_signal = tf.zeros(shape=(1_000_000, 2))
    noisy_signal = channel(zero_signal)
    noise = noisy_signal
    noise_power_channel = tf.reduce_mean(tf.reduce_sum(tf.square(noise), axis=-1))
    noise_power_channel_db = lineartodecibel(noise_power_channel)
    tf.debugging.assert_near(noise_power_channel_db, noise_power_db, atol=1e-2)


def test_awgn_eval_vs_training():
    training_noise_power = tf.constant(1e-4, dtype=tf.float32)
    eval_noise_power = tf.constant(1e-2, dtype=tf.float32)
    channel = SimpleAWGN(
        noise_power=training_noise_power, eval_noise_power=eval_noise_power
    )
    zero_signal = tf.zeros(shape=(1_000_000, 2))
    training_noisy_signal = channel.call(zero_signal, training=True)
    eval_noisy_signal = channel.call(zero_signal, training=False)
    training_noise_power_channel = tf.reduce_mean(
        tf.reduce_sum(tf.square(training_noisy_signal), axis=-1)
    )
    eval_noise_power_channel = tf.reduce_mean(
        tf.reduce_sum(tf.square(eval_noisy_signal), axis=-1)
    )
    tf.debugging.assert_near(
        training_noise_power_channel, training_noise_power, atol=1e-4
    )
    tf.debugging.assert_near(eval_noise_power_channel, eval_noise_power, atol=1e-4)


def test_awgn_noise_power_update():
    noise_power = tf.constant(1e-4, dtype=tf.float32)
    channel = SimpleAWGN(noise_power=noise_power)
    zero_signal = tf.zeros(shape=(1_000_000, 2))
    noisy_signal = channel(zero_signal, training=True)  #!training=True
    noise = noisy_signal
    noise_power_channel = tf.reduce_mean(tf.reduce_sum(tf.square(noise), axis=-1))
    tf.debugging.assert_near(noise_power_channel, noise_power, atol=1e-4)

    new_noise_power = tf.constant(1e-2, dtype=tf.float32)
    channel.set_noise_power(
        eval_noise_power=new_noise_power
    )  #! noise_power/eval_noise_power never used as we call channel without training=False/True
    zero_signal = tf.zeros(shape=(1_000_000, 2))
    noisy_signal = channel(zero_signal, training=False)  #!training=False
    noise = noisy_signal
    noise_power_channel = tf.reduce_mean(tf.reduce_sum(tf.square(noise), axis=-1))
    tf.debugging.assert_near(noise_power_channel, new_noise_power, atol=1e-4)


"""
def test_awgn_noise_power_update_with_model():
    noise_power = tf.constant(1.0, dtype=tf.float32)
    model = SimpleAWGNModel(noise_power=noise_power)
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(1e-1),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    )
    zero_signal = tf.zeros(shape=(1_000, 2))
    noisy_signal = model(zero_signal)
    noise = noisy_signal
    noise_power_channel = tf.reduce_mean(tf.reduce_sum(tf.square(noise), axis=-1))
    tf.debugging.assert_near(noise_power_channel, noise_power, atol=1e-4)

    new_noise_power = tf.constant(2.0, dtype=tf.float32)
    model.channel.set_noise_power(eval_noise_power=new_noise_power)
    zero_signal = tf.zeros(shape=(1_000, 2))
    noisy_signal = model(zero_signal)
    noise = noisy_signal
    noise_power_channel = tf.reduce_mean(tf.reduce_sum(tf.square(noise), axis=-1))
    tf.debugging.assert_near(noise_power_channel, new_noise_power, atol=1e-4)
"""

if __name__ == "__main__":
    pytest.main()
