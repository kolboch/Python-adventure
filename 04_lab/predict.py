# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
#  implemented K. Bochynski
# --------------------------------------------------------------------------

import pickle as pkl
import numpy as np

TRAIN_DATA_FILE_PATH = 'train.pkl'
NUMBER_OF_LABELS = 36


def load_training_data():
    with open(TRAIN_DATA_FILE_PATH, 'rb') as f:
        return pkl.load(f)


def predict(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 35} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """
    x_train, y_train = load_training_data()

    cut = 5000

    x_train = prepare_x(x_train[:cut])
    x = prepare_x(x)
    y_train = y_train[:cut]  # cutting y train set

    distance = hamming_distance(x > 0, x_train > 0)
    index_min_distances = np.argmin(distance, axis=1)  # N x 1
    return y_train[index_min_distances]
    pass


def hamming_distance(x_array, x_train_array):
    return np.absolute(x_array.dot(x_train_array.T - 1) + (x_array - 1).dot(x_train_array.T))
    pass


def prepare_x(x_to_prepare):
    N = x_to_prepare.shape[0]
    x = np.reshape(x_to_prepare, (N, 56, 56))
    x = np.reshape(x[:, 3:-3, 3:-3], (N, 2500))
    return x
