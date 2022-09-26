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
from . import random_messages, random_messages_dataset, random_messages_base_dataset


def test_messages_shape():
    length = 8
    count = 1_000
    messages = random_messages(length, count)
    assert all(tf.shape(messages) == (count, length))


def test_values_range():
    length = 8
    count = 1_000
    messages = random_messages(length, count)
    assert tf.math.reduce_min(messages) == 0
    assert tf.math.reduce_max(messages) == 1
    assert all(m in {0, 1} for m in tf.reshape(messages, shape=(-1,)).numpy())


def test_seed_repeatability():
    seed = 1
    length = 8
    count = 1_000
    messages1 = random_messages(length, count, seed=seed)
    messages2 = random_messages(length, count, seed=seed)
    tf.debugging.assert_equal(messages1, messages2)


def test_random_messages_dataset_type():
    dataset = random_messages_dataset(8, seed=None)
    message = list(dataset.take(1))
    tf.debugging.assert_type(message, tf.float32)


def test_random_messages_dataset_batch_size():
    dataset = random_messages_dataset(8, seed=None)
    dataset.prefetch(10)
    dataset.batch(2)
    it = dataset.as_numpy_iterator()
    message = [next(it) for _ in range(3)]
    tf.debugging.assert_type(message, tf.float32)


def test_random_messages_base_dataset():
    dataset = random_messages_base_dataset(8)
    it = dataset.as_numpy_iterator()
    messages = [next(it)[0] for _ in range(100)]
    ones_count = tf.reduce_sum(messages, axis=-1)
    assert all(tf.reduce_max(ones_count, axis=-1) == 1)


if __name__ == "__main__":
    pytest.main()
