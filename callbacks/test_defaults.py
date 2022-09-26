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

import tempfile

import tensorflow as tf
from . import default_configuration_tensorboard, default_configuration_early_stopping
from . import default_training_callbacks, default_evaluation_callbacks


def test_training_callbacks():
    with tempfile.TemporaryDirectory() as tmpdirname:
        configuration_early_stopping = default_configuration_early_stopping()
        configuration_tensorboard = default_configuration_tensorboard(tmpdirname)
        callbacks = default_training_callbacks(
            configuration_earlystopping=configuration_early_stopping,
            configuration_tensorboard=configuration_tensorboard,
        )
        assert len(callbacks) == 2
        # TODO: run dumb model to execute callbacks


def test_evaluation_callbacks():
    pass


# TODO: run dumb model to execute callbacks

if __name__ == "__main__":
    pytest.main()
