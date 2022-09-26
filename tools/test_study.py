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

from numpy import dtype
import pytest
import tempfile
import os

import pandas as pd
import tensorflow as tf

from .summary import Summary
from .study import create_paths_and_summaries


def test_study_creation():
    with tempfile.TemporaryDirectory() as tmpdirname:

        index_name = "Eb/N0"
        index = tf.range(0, 9, delta=1.0, dtype=tf.float32)
        paths_and_summaries = create_paths_and_summaries(tmpdirname, index_name, index)

        summary_ber = paths_and_summaries.summary_ber
        summary_bler = paths_and_summaries.summary_bler

        assert summary_ber.index.name == index_name
        assert summary_bler.index.name == index_name
        assert all(summary_bler.index.values == index)
        assert all(summary_ber.index.values == index)

        assert os.path.isdir(paths_and_summaries.results_path) is True
        assert os.path.isdir(paths_and_summaries.tensorboard_path) is True
        assert os.path.isdir(paths_and_summaries.models_path) is True
