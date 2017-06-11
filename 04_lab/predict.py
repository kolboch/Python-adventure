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
    y_val = y_train[10000:11000]
    cut = 10000
    x_train = x_train[:cut]  # cutting x train set
    y_train = y_train[:cut]  # cutting y train set

    distance = hamming_distance(x > 0, x_train > 0)
    index_min_distances = np.argmin(distance, axis=1)  # N x 1
    predictions = y_train[index_min_distances]  # N x 1
    error = measure_error(predictions, y_val)
    print('error:{}'.format(error))
    return y_train[index_min_distances]
    pass


def hamming_distance(x_array, x_train_array):
    return np.absolute(x_array.dot(x_train_array.T - 1) + (x_array - 1).dot(x_train_array.T))
    pass


def measure_error(y_predicted, y_true):
    error = 0
    print('predictions size:{}'.format(y_predicted.shape))
    print('y_true size:{}'.format(y_true.shape))
    for i in range(y_predicted.shape[0]):
        if y_predicted[i] != y_true[i]:
            error += 1
    return error / y_predicted.shape[0]


# def show_images(x_set, y_set, indices):
#     for i in range(len(indices)):
#         image = np.reshape(x_set[indices[i]], (56, 56))
#         plt.imshow(image)
#         plt.show()

if __name__ == "__main__":
    x, y = load_training_data()
    predict(x[10000:11000])
