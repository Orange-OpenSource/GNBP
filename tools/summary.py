#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility to output results as csv spreadsheet

Brief: Utility to output results as csv spreadsheet

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

from argparse import ArgumentError
import pandas as pd


class Summary:
    def __init__(self, filepath, index_name=None, index=None):
        """Summary of results output to an excel spreadsheet

        Args:
            filepath (path): path of the output file
            index_name (str): column header for the index
            index (array): index vales

        Raises:
            FileNotFoundError: exception raised if the path does not exist
        """
        self.filepath = filepath
        self.df = None
        try:
            dataframe = pd.read_csv(filepath, index_col=0)
            if index_name is not None and index_name != dataframe.index.name:
                raise ArgumentError(
                    f"{filepath} contains a different index name than provided in arguments ({self.dataframe.index.name} vs {index_name})"
                )
            if any(index != dataframe.index.values):
                raise ArgumentError(
                    f"{filepath} contains different index values than provided in arguments ({self.dataframe.index} vs {index})"
                )
        except FileNotFoundError:
            dataframe = pd.DataFrame(index=index)
            dataframe.index.name = index_name
        self.df = dataframe
        self.df.to_csv(self.filepath)

    def __getitem__(self, *args, **kwargs):
        return self.df.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        self.df.__setitem__(*args, **kwargs)
        self.dataframe.to_csv(self.filepath)

    @property
    def index(self):
        return self.df.index

    @property
    def dataframe(self):
        return self.df
