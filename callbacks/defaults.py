#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Default list of keras callbacks

For training: tensorboard, earlystopping
for evaluation: terminations

Brief: default callbacks used for training and evaluating models

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
import os

from .termination import BatchTerminationCallback


def default_configuration_early_stopping(monitor="loss", patience=200):
    """default configuration ofr the early stopping callback

    Args:
        monitor (str, optional): the metric or loss to monitor. Defaults to 'loss'.
        patience (int, optional): default patience. Defaults to 200.

    Returns:
        dict: early stopping callback configuration
    """
    return {
        "monitor": monitor,
        "patience": patience,
        "mode": "min",
        "restore_best_weights": True,
        "verbose": 1,
    }


def default_configuration_reduce_lr_on_plateau(monitor="loss", factor=0.5, patience=10):
    """default configuration of the reduce lr on plateau callback

    Args:
        monitor (str, optional): the metric or loss to monitor. Defaults to 'loss'.
        factor (float, optional): factor by which the learning rate will be reduced. new_lr = lr * factor. defaults to 0.5
        patience (int, optional): default patience. Defaults to 10.


    Returns:
        dict: early stopping callback configuration
    """
    return {
        "monitor": monitor,
        "factor": factor,
        "patience": patience,
        "verbose": 1,
        "mode": "auto",
        "min_delta": 0.0001,
        "cooldown": 0,
        "min_lr": 1e-4,
    }


def default_configuration_tensorboard(log_dir):
    """default configuration for the tensorboard callback

    Args:
        log_dir (str|path): path of the tensorboard logs directory

    Returns:
        dict: tensorboard callback configuration
    """
    return {
        "log_dir": log_dir,
        "update_freq": "epoch",
        "histogram_freq": 1,
        "write_graph": False,
        "embeddings_freq": 1,
        "profile_batch": 2,
    }


def default_configuration_model_checkpoint(
    filepath=os.path.join("tmp", "checkpoint.tf"),
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1,
):
    """default configuration for the model checkpoint callback

    Args:
        filepath (str, optional): [description]. Defaults to "tmp/checkpoint.tf".
        save_weights_only (bool, optional): [description]. Defaults to True.
        monitor (str, optional): [description]. Defaults to "val_loss".
        mode (str, optional): [description]. Defaults to "min".
        save_best_only (bool, optional): [description]. Defaults to True.
        verbose (int, optional): [description]. Defaults to 1.

    Returns:
        (dict): model checkpoint callback configuration
    """

    return {
        "filepath": filepath,
        "save_weights_only": save_weights_only,
        "monitor": monitor,
        "mode": mode,
        "save_best_only": save_best_only,
        "verbose": verbose,
    }


def default_training_callbacks(
    configuration_earlystopping=None,
    configuration_tensorboard=None,
    configuration_reduce_lr_on_plateau=None,
    configuration_model_checkpoint=None,
):
    """return the list of default callbacks for training

    Args:
        configuration_earlystopping (dict, optional): Arguments of tf.keras.callback.EarlyStopping. Defaults to None.
        configuration_tensorboard (dict, optional): Arguments of tf.keras.callback.Tensorboard. Defaults to None.
        configuration_reduce_lr_on_plateau (dict, optional): Arguments of tf.keras.callback.ReduceLROnPlateau. Defaults to None.
        configuration_model_checkpoint (dict, optional): Arguments of tf.keras.callback.ModelCheckpoint. Defaults to None.

    Returns:
        list: callbacks
    """
    callbacks = []
    if configuration_earlystopping is not None:
        early_stopping = tf.keras.callbacks.EarlyStopping(**configuration_earlystopping)
        callbacks.append(early_stopping)
    if configuration_tensorboard is not None:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            **configuration_tensorboard
        )
        callbacks.append(tensorboard_callback)
    if configuration_reduce_lr_on_plateau is not None:
        reduce_lr_on_plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(
            **configuration_reduce_lr_on_plateau
        )  # TensorBoard(**configuration_tensorboard)
        callbacks.append(reduce_lr_on_plateau_callback)
    if configuration_model_checkpoint is not None:
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            **configuration_model_checkpoint
        )
        callbacks.append(model_checkpoint_callback)
    return callbacks


def default_evaluation_callbacks(configuration_termination=None):
    """list of default evaluation callbacks

    Args:
        configuration_termination (dict, optional): BatchTerminationCallback arguments. Defaults to None.

    Returns:
        list: callbacks
    """
    callbacks = []
    if configuration_termination is not None:
        termination = BatchTerminationCallback(**configuration_termination)
        callbacks.append(termination)
    return callbacks
