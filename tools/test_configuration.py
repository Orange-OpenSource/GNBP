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

from . import configurations_product
from . import configurations_list


def test_configurations_product():
    options = {"a": [1, 2, 3], "b": ["bob", "alice"]}
    configurations = [c for c in configurations_product(**options)]
    assert len(configurations) == 6
    assert configurations[0].a == 1
    assert configurations[5].b == "alice"


def test_configurations_list():
    options_list = ["a", "b"]
    config_list = [[1, "bob"], [3, "alice"]]
    configurations = configurations_list(options_list, config_list)
    assert len(configurations) == 2
    assert configurations[0].a == 1
    assert configurations[1].b == "alice"


if __name__ == "__main__":
    pytest.main()
