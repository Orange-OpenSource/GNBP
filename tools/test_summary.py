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
import os

import pandas as pd

from .summary import Summary


def test_summary_csv_file_creation():
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = os.path.join(tmpdirname, "test.csv")
        summary = Summary(
            filepath=filepath, index_name="index_name", index=[1.1, 1.2, 1.3, 1.4]
        )
        assert os.path.isfile(filepath) is True


def test_summary_csv_file_index():
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = os.path.join(tmpdirname, "test.csv")
        index = [1.1, 1.2, 1.3, 1.4]
        index_name = "index_name"
        summary = Summary(
            filepath=filepath, index_name="index_name", index=[1.1, 1.2, 1.3, 1.4]
        )
        df = pd.read_csv(filepath, index_col=0)
        assert df.index.name == index_name
        assert all(index == df.index.values)
        assert summary.index.name == index_name
        assert all(summary.index.values == index)


def test_summary_csv_file_input():
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = os.path.join(tmpdirname, "test.csv")
        index = [1.1, 1.2, 1.3, 1.4]
        index_name = "index_name"
        summary = Summary(
            filepath=filepath, index_name="index_name", index=[1.1, 1.2, 1.3, 1.4]
        )
        summary["model"] = [2, 3, 4, 5]

        df = pd.read_csv(filepath, index_col=0)
        assert all(df["model"] == [2, 3, 4, 5])


if __name__ == "__main__":
    pytest.main()
