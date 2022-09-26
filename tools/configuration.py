#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility to manage experiment configurations

Brief: Utility to manage experiment configurations

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

from itertools import product, starmap
from collections import namedtuple


def configurations_product(**items):
    """generate an iterable of configurations obtained from the products of named arguments

    Returns:
        Configuration: a configuration built from named arguments
    """
    Configuration = namedtuple("Configuration", items.keys())
    return starmap(Configuration, product(*items.values()))


def configurations_list(options_list, config_list):
    """generates an iterable of configuration given a list of options and a list of configuration list
    corresponding to listed options. e.g. given list option ["a,"b"] and configuration list [[1,2],[3,4]]
    the function returns [Configuration("a":1,"b":2),Configuration("a":3,"b":4)]

    Args:
        options_list ([string]): list of options
        configurations_list ([[undefined]]): list of list of associated configuration parameters

    Returns:
        [namedtuple Configuration]: list of configurations
    """
    configurations = []
    Configuration = namedtuple("Configuration", options_list)
    for c in config_list:
        configurations.append(Configuration(*c))

    return configurations
