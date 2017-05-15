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
import matplotlib.pyplot as plt

TRAIN_DATA_FILE_PATH = 'train.pkl'


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

    my_print('x shape', x.shape)
    pass


def my_print(label, value):
    print('{}: {}'.format(label, value))


def run_training():
    train_x, train_y = load_training_data()
    predict(train_x)
    indices_to_show = [1, 23, 77, 109, 200]
    show_images(train_x, indices_to_show)


def show_images(x_set, indices):
    for i in range(len(indices)):
        image = np.reshape(x_set[indices[i]], (56, 56))
        plt.imshow(image)
        plt.show()


if __name__ == "__main__":
    run_training()
