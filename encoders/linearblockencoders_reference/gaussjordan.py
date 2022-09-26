#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Brief: Compute a generator matrix from parity check matrix thanks to the Gauss Jordan method.
Author: Louis-Adrien DUFRENE <louisadrien.dufrene@orange.com>
License: MPL-v2.0

"""
import tensorflow as tf
import numpy as np



class GaussJordanGF2(tf.keras.layers.Layer):
    """
    Compute a generator matrix from parity check matrix thanks to the Gauss Jordan method.
    The parity check matrix should be singular for the method to work.
    If the parity check matrix is non singular, no error is raised
    """
    def __init__(self,
                n,
                k,
                output_all=False,
                trainable=False,
                **kwargs):
        
        super(GaussJordanGF2, self).__init__(**kwargs)

        self.n = n
        self.k = k
        self.n_paritycheck = n-k
        self.output_all = output_all
        self.trainable = trainable

    def build(self, _):
        pass

    def call(self, inputs):
        # Rename for code readability
        H_input = inputs
        nb_row = self.n_paritycheck
        nb_col = self.n

        col_swap = tf.range(nb_col, dtype=tf.int32) ## variable used to store the swap operations on columns

        # First step: apply linear combinations on rows of H to make it systematic, we use rows as pivot
        # The diagonal of ones is usually on the right side of H
        # Legacy behavior constructed the eye on the left side of H, see LegacyGaussJordanGF2() layer
        pivot_index = nb_row
        for col_index in tf.range(start=nb_col-1, limit=-1, delta=-1, dtype=tf.int32): ## start the construction from the last column
            if pivot_index != 0: ## if pivot_index==0, all the rows were parsed, the eye columns should be available within H
                max_row_index = (pivot_index - 1) - tf.math.argmax(H_input[pivot_index-1::-1, col_index], output_type=tf.int32) ## from the pivot row to the first row, gives the first row index with a 1

                # Check if current pivot value is 1, otherwise there was no 1 available in the column
                if H_input[max_row_index, col_index] != 0:
                    pivot_index -= 1 ## current pivot index

                    # Swap the rows to place the 1 at the pivot position
                    if max_row_index != pivot_index:
                        H_input = tf.tensor_scatter_nd_update(H_input, [[max_row_index],[pivot_index]], tf.gather(H_input, [pivot_index, max_row_index], axis=0))
                    
                    # XOR the rows which have a 1 in the current column, with the pivot row to construct the current eye column
                    pivot_row = tf.expand_dims(H_input[pivot_index, :], axis=0) ## get the pivot row
                    pivot_col = H_input[:, col_index] ## get the current column
                    pivot_col = tf.expand_dims(tf.tensor_scatter_nd_update(pivot_col, [[pivot_index]], [0]), axis=1) ## set the pivot row to zero to avoid self xor
                    H_update = tf.matmul(pivot_col, pivot_row) ## if the column element is 0, the row the null => no xor operation, otherwise the row will be used to xor
                    H_input = tf.math.mod(tf.math.add(H_input, H_update), 2) ## resulting xor operation

        # Second step: the element of the eye should be present within H
        # The columns are swap to place the eye on the right side of H
        base_vector = tf.eye(nb_row, dtype=tf.float32)
        for identity_index in tf.range(nb_row, dtype=tf.int32):
            identity_check = tf.reduce_sum(tf.abs(H_input - tf.expand_dims(base_vector[:, identity_index], axis=1)), axis=0)
            min_id_check = tf.reduce_min(identity_check) ## check that at least one column is equal to the current eye column
            
            if min_id_check == 0: ## the column is present in H
                argmin_id_check = tf.math.argmin(identity_check, output_type=tf.int32) ## get the column index
                current_eye_index = nb_col - nb_row + identity_index

                if argmin_id_check != current_eye_index: ## if False, need to swap column to construct the eye
                    # Scatter and update slicing is easy to use on rows, so we need to transpose the matrix before using it for column
                    H_input = tf.transpose(H_input)
                    H_input = tf.tensor_scatter_nd_update(H_input, [[argmin_id_check],[current_eye_index]], tf.gather(H_input, [current_eye_index, argmin_id_check], axis=0))
                    H_input = tf.transpose(H_input)
                    # Store the index swapped
                    col_swap = tf.tensor_scatter_nd_update(col_swap, [[argmin_id_check],[current_eye_index]], tf.gather(col_swap, [current_eye_index, argmin_id_check]))

        # Third step: construct the generator matrix G from the parity part of H
        H_parity = H_input[:, 0:self.k] ## get the parity part of H, so everything but the eye

        G_sys = tf.concat([tf.eye(self.k, dtype=tf.float32), tf.transpose(H_parity)], axis=1) # Construct the systematic generator matrix with H parity transpose

        # To compute the final generator matrix, we need to apply the stored column
        G_nsys = tf.transpose(G_sys)
        G_nsys = tf.tensor_scatter_nd_update(G_nsys, tf.expand_dims(col_swap, axis=1), G_nsys)
        G_nsys = tf.transpose(G_nsys)

        src_index = col_swap[0:self.k] # the systematic bits output position after endoding
        par_index = col_swap[self.k:self.n] # the parity bits output position after encoding
        
        if self.output_all:
            return G_nsys, G_sys, H_input, src_index, par_index
        else:
            return G_nsys, src_index

def gen_code(H_nsys,n,k):
    GJ_layer = GaussJordanGF2(n, k, output_all=True)
    (G_nsys, G_sys, H_sys, src_index, par_index) = GJ_layer(H_nsys)
    G_nsys = np.array(G_nsys)
    G_sys = np.array(G_sys)
    H_sys = np.array(H_sys)
    return (H_nsys,H_sys,G_nsys,G_sys)