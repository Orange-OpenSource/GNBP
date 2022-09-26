#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Default paths for this project

results are stored in `./{study-name}/results`
tensorboard are in `./{study-name}/tensorboard`
models are in stored `./{study-name}/models`

Brief: default paths used in this project

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

import os

from collections import namedtuple

from . import Summary


def create_paths_and_summaries(path, index_name, index):

    results_path = os.path.join(path, "results")
    tensorboard_path = os.path.join(path, "tensorboard")
    models_path = os.path.join(path, "models")

    summary_ber_path = os.path.join(results_path, "summary-ber.csv")
    summary_bler_path = os.path.join(results_path, "summary-bler.csv")
    summary_bec_path = os.path.join(results_path, "summary-bec.csv")
    summary_blec_path = os.path.join(results_path, "summary-blec.csv")
    summary_bpci_ber_path = os.path.join(results_path, "summary-bpci-ber.csv")
    summary_bpci_bler_path = os.path.join(results_path, "summary-bpci-bler.csv")

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    summary_ber = Summary(summary_ber_path, index_name=index_name, index=index)
    summary_bler = Summary(summary_bler_path, index_name=index_name, index=index)
    summary_bec = Summary(summary_bec_path, index_name=index_name, index=index)
    summary_blec = Summary(summary_blec_path, index_name=index_name, index=index)
    summary_bpci_ber = Summary(
        summary_bpci_ber_path, index_name=index_name, index=index
    )
    summary_bpci_bler = Summary(
        summary_bpci_bler_path, index_name=index_name, index=index
    )

    paths_and_summaries = namedtuple(
        "PathsAndSummaries",
        [
            "results_path",
            "tensorboard_path",
            "models_path",
            "summary_ber",
            "summary_bler",
            "summary_bec",
            "summary_blec",
            "summary_bpci_ber",
            "summary_bpci_bler",
        ],
    )
    paths_and_summaries.results_path = results_path
    paths_and_summaries.tensorboard_path = tensorboard_path
    paths_and_summaries.models_path = models_path
    paths_and_summaries.summary_ber = summary_ber
    paths_and_summaries.summary_bler = summary_bler
    paths_and_summaries.summary_bec = summary_bec
    paths_and_summaries.summary_blec = summary_blec
    paths_and_summaries.summary_bpci_ber = summary_bpci_ber
    paths_and_summaries.summary_bpci_bler = summary_bpci_bler

    return paths_and_summaries
