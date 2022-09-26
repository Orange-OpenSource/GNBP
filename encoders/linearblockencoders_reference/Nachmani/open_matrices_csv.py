import csv
import numpy as np


def open_PC_matrix(csv_path):
    print(csv_path)
    with open(csv_path) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=";")
        H = []
        for row in csv_reader:
            H.append(row)
        H = np.array(H, dtype=np.uint8)
    # print(H)
    # n = np.shape(H)[1]
    # print(n)
    # k = np.shape(H)[0]
    # print(k)
    # delta = np.mean(H)
    # print(delta)
    return H
