#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Random messages dataset

Brief: random messages dataset

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
import numpy as np


def random_messages(length, count, seed=None):
    """generate `count` random messages of length `length`.

    Args:
        length (int): length of the messages in bits
        count (int): number of messages to generate
        seed (integer, optional): RNG seed. Defaults to None.

    Returns:
        tf.Tensor(dtype=tf.float3): random messages
    """
    if seed is None:
        rng = tf.random.get_global_generator()
    else:
        rng = tf.random.Generator.from_seed(seed)
    dataset = tf.cast(
        rng.uniform((count, length), maxval=2, dtype=tf.int32), dtype=tf.float32
    )
    return dataset


def random_messages_dataset(length, batch=256, prefetch=1024 ** 2, seed=None):
    """generate random sequences of bits of length `length` as dataset

    Args:
        length (tf.int32): length of bits sequences
        batch (tf.int32, optional): batch size. Defaults to 256.
        prefetch (tf.int32, optional): count of pre-generated sequences. Defaults to 1024**2.
        seed (tf.int64|none, optional): generator's seed. Defaults to None. If None, retrieve global generator.

    Returns:
        tf.data.Dataset: dataset
    """
    if seed is None:
        rng = tf.random.get_global_generator()
    else:
        rng = tf.random.Generator.from_seed(seed)

    try:
        AUTOTUNE = tf.data.AUTOTUNE
    except:
        AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataset = (
        tf.data.Dataset.from_tensors(tf.zeros(shape=(length,), dtype=tf.float32))
        .repeat(count=None)
        .batch(batch)  # Apply batch before mapping for vectorizing the mapping
        .map(
            lambda x: x
            + tf.cast(
                rng.uniform(shape=tf.shape(x), minval=0, maxval=2, dtype=tf.int32),
                dtype=tf.float32,
            ),
            num_parallel_calls=AUTOTUNE,
        )
        .map(lambda x: (x, x), num_parallel_calls=AUTOTUNE)
        .prefetch(prefetch)
    )
    return dataset


def random_messages_base_dataset(length, batch=256, prefetch=1024 ** 2, seed=None):
    """generate random bases of the sequences of bits of length `length`, e.g. [0,...,0,1,0,...,0] as dataset

    Args:
        length (tf.int32): length of bits sequences
        batch (tf.int32, optional): batch size. Defaults to 256.
        prefetch (tf.int32, optional): count of pre-generated sequences. Defaults to 1024**2.
        seed (tf.int64|none, optional): generator's seed. Defaults to None. If None, retrieve global generator.

    Returns:
        [type]: [description]
    """
    if seed is None:
        rng = tf.random.get_global_generator()
    else:
        rng = tf.random.Generator.from_seed(seed)
    base = tf.eye(length, dtype=tf.float32)

    try:
        AUTOTUNE = tf.data.AUTOTUNE
    except:
        AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataset = (
        tf.data.Dataset.from_tensors(tf.zeros(shape=(length,), dtype=tf.float32))
        .repeat(count=None)
        .batch(batch)  # Apply batch before mapping for vectorizing the mapping
        .map(
            lambda x: x
            + tf.gather(
                base,
                rng.uniform(
                    shape=(1, tf.shape(x)[0]), minval=0, maxval=length, dtype=tf.int32
                ),
            )[0],
            num_parallel_calls=AUTOTUNE,
        )
        .map(lambda x: (x, x), num_parallel_calls=AUTOTUNE)
        .prefetch(prefetch)
    )
    return dataset


def all_zero_dataset(length, batch=256, prefetch=1024 ** 2):
    """generate zero sequences of length `length` as dataset

    Args:
        length (tf.int32): length of bits sequences
        batch (tf.int32, optional): batch size. Defaults to 256.
        prefetch (tf.int32, optional): count of pre-generated sequences. Defaults to 1024**2.

    Returns:
        [type]: [description]
    """
    try:
        AUTOTUNE = tf.data.AUTOTUNE
    except:
        AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataset = (
        tf.data.Dataset.from_tensors(tf.zeros(shape=(length,), dtype=tf.float32))
        .repeat(count=None)
        .batch(batch)  # Apply batch before mapping for vectorizing the mapping
        .map(lambda x: x, num_parallel_calls=AUTOTUNE)
        .map(lambda x: (x, x), num_parallel_calls=AUTOTUNE)
        .prefetch(prefetch)
    )
    return dataset


def random_messages_base_all_zero_all_one_dataset(
    length, batch=256, prefetch=1024 ** 2, seed=None
):
    """generate random bases of the sequences of bits of length `length`, e.g. [0,...,0,1,0,...,0]
    also including the all-zero and all-one sequences as dataset

    Args:
        length (tf.int32): length of bits sequences
        batch (tf.int32, optional): batch size. Defaults to 256.
        prefetch (tf.int32, optional): count of pre-generated sequences. Defaults to 1024**2.
        seed (tf.int64|none, optional): generator's seed. Defaults to None. If None, retrieve global generator.

    Returns:
        [type]: [description]
    """
    if seed is None:
        rng = tf.random.get_global_generator()
    else:
        rng = tf.random.Generator.from_seed(seed)
    words = tf.concat(
        [
            tf.zeros(shape=[1, length]),
            tf.eye(length, dtype=tf.float32),
            tf.ones(shape=[1, length]),
        ],
        axis=0,
    )

    try:
        AUTOTUNE = tf.data.AUTOTUNE
    except:
        AUTOTUNE = tf.data.experimental.AUTOTUNE

    dataset = (
        tf.data.Dataset.from_tensors(tf.zeros(shape=(length,), dtype=tf.float32))
        .repeat(count=None)
        .batch(batch)  # Apply batch before mapping for vectorizing the mapping
        .map(
            lambda x: x
            + tf.gather(
                words,
                rng.uniform(
                    shape=(1, tf.shape(x)[0]),
                    minval=0,
                    maxval=length + 2,
                    dtype=tf.int32,
                ),
            )[0],
            num_parallel_calls=AUTOTUNE,
        )
        .map(lambda x: (x, x), num_parallel_calls=AUTOTUNE)  # .cache()
        .prefetch(prefetch)
    )
    return dataset
