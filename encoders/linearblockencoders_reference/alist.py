"""
Tools to convert alist code files to npz array format

Brief: Tools to convert alist code files to npz array format compatible with the other tools of this repository.

Copyright (c) 2022 Guillaume Larue <guillaume.larue@orange.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice (including the next paragraph) shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS 
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
"""

import numpy as np


def gf2elim(M):

    m, n = M.shape

    i = 0
    j = 0

    while i < m and j < n:
        # find value and index of largest element in remainder of column j
        k = np.argmax(M[i:, j]) + i

        # swap rows
        # M[[k, i]] = M[[i, k]] this doesn't work with numba
        temp = np.copy(M[k])
        M[k] = M[i]
        M[i] = temp

        aijn = M[i, j:]

        col = np.copy(M[:, j])  # make a copy otherwise M will be directly affected

        col[i] = 0  # avoid xoring pivot row with itself

        flip = np.outer(col, aijn)

        M[:, j:] = (M[:, j:] ^ flip) % 2

        i += 1
        j += 1

    return M


def a_list_to_parity_check(a_list):
    n = a_list[0][0]
    m = a_list[0][1]

    var_idx = a_list[4 : 4 + n]
    check_idx = a_list[4 + n : 4 + n + 1 + m]

    parity_check_matrix = np.zeros([n, m])
    parity_check_matrix_2 = np.zeros([n, m])

    # V1
    for i in range(n):
        for idx in var_idx[i]:
            if idx != 0:
                parity_check_matrix[i][(idx - 1)] = 1

    # V2
    for i in range(m):
        for idx in check_idx[i]:
            if idx != 0:
                parity_check_matrix_2[(idx - 1)][i] = 1

    return parity_check_matrix


def generate_code(a_list_path):
    with open(a_list_path, "r") as myfile:
        data = myfile.readlines()
        a_list = [
            [int(x) for x in elt.replace(" \n", "").replace(" ", ",").split(",")]
            for elt in data
        ]

    n = a_list[0][0]
    k = n - a_list[0][1]

    H_non_systematic = np.array(a_list_to_parity_check(a_list).T, dtype=np.uint8)

    H_systematic = np.array(gf2elim(np.copy(H_non_systematic)), dtype=np.float32)

    redundancy_H = np.array([H_systematic[i][n - k :] for i in range(n - k)])

    H_systematic = np.concatenate((redundancy_H, np.eye(n - k)), axis=-1)

    G = np.concatenate((np.eye(k), redundancy_H.T), axis=-1)

    redundancy_G = np.array([G[i][k:] for i in range(k)])

    return (
        H_non_systematic.astype(np.float32),
        H_systematic,
        G,
        redundancy_H,
        redundancy_G,
    )


def save_as_npz(code_name="BCH_7_4", main_save_path="./"):
    a_list_path = main_save_path + code_name + "_alist.txt"
    H_non_systematic, H_systematic, G, redundancy_H, redundancy_G = generate_code(
        a_list_path
    )

    npz_path = main_save_path + code_name + ".npz"
    np.savez(
        npz_path,
        G=G,
        H_systematic=H_systematic,
        H_non_systematic=H_non_systematic,
        redundancy_H=redundancy_H,
        redundancy_G=redundancy_G,
    )
