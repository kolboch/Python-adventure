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
import time

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

    pass


def my_print(label, value):
    print('{}: {}'.format(label, value))


def run_training():
    x_set, y_set = load_training_data()
    predict(x_set)
    # indices_to_show = [1, 23, 77, 109, 200]
    # show_images(x_set, y_set, indices_to_show)

    # wyznaczyc zbior treningowy ok 90% całego zbioru
    my_print('train x shape', x_set.shape)
    x_train = x_set[:int(x_set.shape[0] / 10 * 9)]
    y_train = y_set[:x_train.shape[0]]
    my_print('x_train shape', x_train.shape)
    my_print('y_train shape', y_train.shape)
    # zbiory walidacyjne to co zostalo
    x_val = x_set[x_train.shape[0]:]
    y_val = y_set[y_train.shape[0]:]
    my_print('x val shape', x_val.shape)
    my_print('y val shape', y_val.shape)

    # making prediction, for knn no learning required just matching with what we got
    start = time.time()
    my_print('knn starting', start)

    k_test = 55  # number o k nearest neighbours considered
    # x_train = x_train[:3000]
    # y_train = y_train[:3000]
    # error for 3000, 2752
    distance = hamming_distance(x_val, x_train)
    sorted_labels = sort_train_labels_knn(distance, y_train)

    p_y_x_distribution = p_y_x_knn(sorted_labels, k_test)

    error_knn = classification_error(p_y_x_distribution, y_val)
    print('error', error_knn)

    end = time.time()
    my_print('time elapsed', end - start)


def show_images(x_set, y_set, indices):
    for i in range(len(indices)):
        image = np.reshape(x_set[indices[i]], (56, 56))
        plt.imshow(image)
        my_print('y label', y_set[i])
        plt.show()


# knn methods

def hamming_distance(x, x_train):
    result = np.absolute(x.dot(x_train.T - 1) + (x - 1).dot(x_train.T))
    my_print('hamming distance', 'checked')
    return result


def sort_train_labels_knn(distances, y):
    w = distances.argsort(kind='mergesort')
    my_print('sort train labels', 'checked')
    return y[w]


def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszych sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """
    # resized = np.delete(y, range(k, y.shape[1]), axis=1)
    # output = np.vstack(np.apply_along_axis(np.bincount, axis=1, arr=resized, minlength=NUMBER_OF_LABELS + 1))
    # output = np.delete(output, 0, axis=1)
    # output = np.divide(output, k)
    result = np.empty([y.shape[0], NUMBER_OF_LABELS])
    for i in range(y.shape[0]):
        for j in range(NUMBER_OF_LABELS):
            sum_for_class = 0
            for x in range(k):
                if y[i, x] == j + 1:
                    sum_for_class += 1
            result[i, j] = sum_for_class
    my_print('p_y_x_knn', 'checked')
    return result / k


def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """
    p_y_x = np.fliplr(p_y_x)
    y_truea = p_y_x.shape[1] - np.argmax(p_y_x, axis=1)
    y_truea = np.subtract(y_truea, y_true)
    diff = np.count_nonzero(y_truea)
    diff /= y_true.shape[0]
    print('classification error', 'checked')
    return diff


def model_selection_knn(x_val, x_train, y_val, y_train, k_values):
    """
    :param x_val: zbior danych walidacyjnych N1xD
    :param x_train: zbior danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors), gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, errors - lista wartosci bledow dla kolejnych k z k_values
    """
    distance = hamming_distance(x_val, x_train)
    sorted_labels = sort_train_labels_knn(distance, y_train)
    min_error = float('inf')
    min_k_index = -1
    k_errors = np.empty(len(k_values))
    for i in range(len(k_values)):
        p_y_x_distribution = p_y_x_knn(sorted_labels, k_values[i])
        current_k_error = classification_error(p_y_x_distribution, y_val)
        if current_k_error < min_error:
            min_error = current_k_error
            min_k_index = i
        k_errors[i] = current_k_error
    return min_error, k_values[min_k_index], k_errors


if __name__ == "__main__":
    run_training()
