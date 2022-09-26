"""Clipping weights constraint

Brief: Keras weights constraint to clip model weights when above/below certain threshold

Copyright (c) 2022 Orange

Author: Guillaume Larue <guillaume.larue@orange.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice (including the next paragraph) shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
"""

import tensorflow as tf


class ClipByValueConstraint(tf.keras.constraints.Constraint):
    def __init__(self, clip_value_min=0.0, clip_value_max=1.0, **kwargs):
        super(ClipByValueConstraint, self).__init__(**kwargs)
        self.clip_value_min = clip_value_min
        self.clip_value_max = clip_value_max

    def __call__(self, w):
        return tf.clip_by_value(
            w, clip_value_min=self.clip_value_min, clip_value_max=self.clip_value_max
        )
