# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division
import numpy as np
import numpy.core


def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. ODleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """
    x_array = X.toarray()
    x_train_array = X_train.toarray()
    return np.absolute(x_array.dot(x_train_array.T - 1) + (x_array - 1).dot(x_train_array.T))
    pass


def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2 (array)
    :param y: wektor etykiet o dlugosci N2 (array)
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """
    result_matrix = np.empty(Dist.shape)
    for i in range(Dist.shape[0]):
        sort_indices = np.argsort(Dist[i], axis=-1, kind='mergesort')
        for j in range(len(y)):
            result_matrix[i, j] = y[sort_indices[j]]
    return result_matrix
    pass


def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszych sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """
    number_of_classes = 4
    result = np.empty([y.shape[0], number_of_classes])
    for i in range(y.shape[0]):
        for j in range(number_of_classes):
            sum_for_class = 0
            for x in range(k):
                if y[i, x] == j + 1:
                    sum_for_class += 1
            result[i, j] = sum_for_class
    return result / k
    pass


def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """
    error_sum = 0
    for i in range(len(y_true)):
        selected_class = choose_max_label(p_y_x[i])
        if y_true[i] != selected_class:
            error_sum += 1
    return error_sum / len(y_true)
    pass


def choose_max_label(values):
    max_current = -1  # only minus one as we deal with probabilites it's enough
    min_index = -1
    for i in range(len(values)):
        if values[i] >= max_current:
            min_index = i
            max_current = values[i]
    return min_index + 1 #as we choose max label, which is index + 1
    pass


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    :param X_val: zbior danych walidacyjnych N1xD
    :param X_train: zbior danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors), gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, errors - lista wartosci bledow dla kolejnych k z k_values
    """
    distance = hamming_distance(X_val, X_train)
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
    pass


def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: etykiety dla dla danych treningowych 1xN
    :return: funkcja wyznacza rozklad a priori p(y) i zwraca p_y - wektor prawdopodobienstw a priori 1xM
    """
    unique, counts = numpy.unique(ytrain, return_counts=True)
    labels_occurrences_dict = dict(zip(unique, counts))
    result_vector = np.empty(len(labels_occurrences_dict))
    for i in range(len(labels_occurrences_dict)):
        result_vector[i] = labels_occurrences_dict[i + 1] / len(ytrain)
    return result_vector
    pass


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: dane treningowe NxD
    :param ytrain: etykiety klas dla danych treningowych 1xN
    :param a: parametr a rozkladu Beta
    :param b: parametr b rozkladu Beta
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(x|y) zakladajac, ze x przyjmuje wartosci binarne i ze elementy
    x sa niezalezne od siebie. Funkcja zwraca macierz p_x_y o wymiarach MxD.
    """
    classes_nr = 4

    unique, counts = numpy.unique(ytrain, return_counts=True)
    labels_occurrences_dict = dict(zip(unique, counts))
    result_matrix = np.zeros([classes_nr, Xtrain.shape[1]])

    for j in range(Xtrain.shape[1]):
        for i in range(Xtrain.shape[0]):
            if Xtrain[i, j]:
                result_matrix[ytrain[i] - 1, j] += 1
    result_matrix = result_matrix + a - 1
    for i in range(classes_nr):
        result_matrix[i] /= (labels_occurrences_dict[i + 1] + a + b - 2)
    return result_matrix
    pass


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.
    """
    n_m_apriori = np.tile(p_y, (X.shape[0], 1))
    for k in range(len(p_y)):
        for row in range(n_m_apriori.shape[0]):
            prob_row_x = 1
            for d in range(X.shape[1]):
                if X[row, d]:
                    p_x_distribution = p_x_1_y[k, d]
                else:
                    p_x_distribution = 1 - p_x_1_y[k, d]
                prob_row_x *= p_x_distribution
            prob_row_x /= X.shape[1]
            n_m_apriori[row, k] *= prob_row_x

    for row in range(n_m_apriori.shape[0]):
        n_m_apriori[row] /= np.sum(n_m_apriori[row])

    return n_m_apriori
    pass


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: zbior danych treningowych N2xD
    :param Xval: zbior danych walidacyjnych N1xD
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrow a do sprawdzenia
    :param b_values: lista parametrow b do sprawdzenia
    :return: funkcja wykonuje selekcje modelu Naive Bayes - wybiera najlepsze wartosci parametrow a i b. Funkcja zwraca
    krotke (error_best, best_a, best_b, errors) gdzie best_error to najnizszy
    osiagniety blad, best_a - a dla ktorego blad byl najnizszy, best_b - b dla ktorego blad byl najnizszy,
    errors - macierz wartosci bledow dla wszystkich par (a,b)
    """
    min_error = float('inf')
    min_a_b_index = (-1, -1)
    errors = np.empty([len(a_values), len(b_values)])
    for i in range(len(a_values)):
        for j in range(len(b_values)):
            result_distribution_p_x_y = estimate_p_x_y_nb(Xtrain, ytrain, a_values[i], b_values[j])
            p_y_x = p_y_x_nb(estimate_a_priori_nb(ytrain), result_distribution_p_x_y, Xval)
            current_error = classification_error(p_y_x, yval)
            errors[i, j] = current_error
            if current_error < min_error:
                min_error = current_error
                min_a_b_index = (i, j)
    return min_error, a_values[min_a_b_index[0]], b_values[min_a_b_index[1]], errors
    pass
