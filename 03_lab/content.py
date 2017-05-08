# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 3: Regresja logistyczna
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
#
# implemented by K. Bochyński
# --------------------------------------------------------------------------

import numpy as np
from sklearn import metrics
from functools import partial


def sigmoid(x):
    '''
    :param x: wektor wejsciowych wartosci Nx1
    :return: wektor wyjściowych wartości funkcji sigmoidalnej dla wejścia x, Nx1
    '''
    return 1 / (1 + np.exp(-x))
    pass


def logistic_cost_function(w, x_train, y_train):
    '''
    :param w: parametry modelu Mx1
    :param x_train: ciag treningowy - wejscia NxM
    :param y_train: ciag treningowy - wyjscia Nx1
    :return: funkcja zwraca krotke (val, grad), gdzie val oznacza wartosc funkcji logistycznej, a grad jej gradient po w
    '''
    n_size = x_train.shape[0]
    m_size = x_train.shape[1]
    val = 0
    grad = np.zeros((m_size, 1))

    for i in range(n_size):
        x_i = x_train[i]
        sigmoid_value = sigmoid(np.dot(w.T, x_i))
        y_i = y_train[i]
        val += (y_i * np.log(sigmoid_value) +
                (1 - y_i) * np.log(1 - sigmoid_value))
        grad += np.reshape(x_i.T * (y_i - sigmoid_value), (m_size, 1))
    val /= -n_size
    grad /= -n_size

    return val[0], grad
    pass


def gradient_descent(obj_fun, w0, epochs, eta):
    '''
    :param obj_fun: funkcja celu, ktora ma byc optymalizowana. Wywolanie val,grad = obj_fun(w).
    :param w0: punkt startowy Mx1
    :param epochs: liczba epok / iteracji algorytmu
    :param eta: krok uczenia
    :return: funkcja wykonuje optymalizacje metoda gradientu prostego dla funkcji obj_fun. Zwraca krotke (w,func_values),
    gdzie w oznacza znaleziony optymalny punkt w, a func_valus jest wektorem wartosci funkcji [epochs x 1] we wszystkich krokach algorytmu
    '''

    w = w0
    func_values = []

    for k in range(epochs):
        val, delta_w = obj_fun(w)
        func_values.append(val)
        w += eta * -delta_w

    func_values.__delitem__(0)
    val, _ = obj_fun(w)
    func_values.append(val)

    return w, np.reshape(np.array(func_values), (epochs, 1))
    pass


def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    '''
    :param obj_fun: funkcja celu, ktora ma byc optymalizowana. Wywolanie val,grad = obj_fun(w,x,y), gdzie x,y oznaczaja podane
    podzbiory zbioru treningowego (mini-batche)
    :param x_train: dane treningowe wejsciowe NxM
    :param y_train: dane treningowe wyjsciowe Nx1
    :param w0: punkt startowy Mx1
    :param epochs: liczba epok
    :param eta: krok uczenia
    :param mini_batch: wielkosc mini-batcha
    :return: funkcja wykonuje optymalizacje metoda stochastycznego gradientu prostego dla funkcji obj_fun. Zwraca krotke (w,func_values),
    gdzie w oznacza znaleziony optymalny punkt w, a func_values jest wektorem wartosci funkcji [epochs x 1] we wszystkich krokach algorytmu. Wartosci
    funkcji do func_values sa wyliczane dla calego zbioru treningowego!
    '''
    w = w0
    func_values = []

    number_of_mini_batches = int(x_train.shape[0] / mini_batch)  # assumption that train samples % mini_batch_size is 0
    batch_size = mini_batch

    batches_x = []
    batches_y = []
    for batch_i in range(number_of_mini_batches):
        batch_begin = batch_i * batch_size
        batches_x.append(x_train[batch_begin:batch_begin + batch_size])
        batches_y.append(y_train[batch_begin:batch_begin + batch_size])

    for k in range(epochs):
        for batch_i in range(number_of_mini_batches):
            _, delta_w = obj_fun(w, batches_x[batch_i], batches_y[batch_i])
            w += eta * -delta_w
        val, _ = obj_fun(w, x_train, y_train)
        func_values.append(val)

    return w, np.reshape(np.array(func_values), (epochs, 1))
    pass


def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    '''
    :param w: parametry modelu Mx1
    :param x_train: ciag treningowy - wejscia NxM
    :param y_train: ciag treningowy - wyjscia Nx1
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (val, grad), gdzie val oznacza wartosc funkcji logistycznej z regularyzacja l2,
    a grad jej gradient po w
    '''
    n_size = x_train.shape[0]
    m_size = x_train.shape[1]
    val = 0
    grad = np.zeros((m_size, 1))

    for i in range(n_size):
        x_i = x_train[i]
        sigmoid_value = sigmoid(np.dot(w.T, x_i))
        y_i = y_train[i]
        val += (y_i * np.log(sigmoid_value) +
                (1 - y_i) * np.log(1 - sigmoid_value))
        grad += np.reshape(x_i.T * (y_i - sigmoid_value), (m_size, 1))

    regularization_val = regularization_lambda / 2 * np.sum(w[1:] ** 2)  # sum without index 0
    w_copy = np.copy(w)
    # now for gradient we omit first index in multiplication, we need copy else change affects val regularization
    w_copy[0] = 0
    regularization_grad = w_copy * regularization_lambda
    val /= -n_size
    grad /= -n_size

    return val[0] + regularization_val, grad + regularization_grad
    pass


def prediction(x, w, theta):
    '''
    :param x: macierz obserwacji NxM
    :param w: wektor parametrow modelu Mx1
    :param theta: prog klasyfikacji z przedzialu [0,1]
    :return: funkcja wylicza wektor y o wymiarach Nx1. Wektor zawiera wartosci etykiet ze zbioru {0,1} dla obserwacji z x
     bazujac na modelu z parametrami w oraz progu klasyfikacji theta
    '''
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = 1 if sigmoid(w.T @ x[i]) > theta else 0
    return y.astype(int).reshape(x.shape[0], 1)
    pass


def f_measure(y_true, y_pred):
    '''
    :param y_true: wektor rzeczywistych etykiet Nx1
    :param y_pred: wektor etykiet przewidzianych przed model Nx1
    :return: funkcja wylicza wartosc miary F
    '''
    return metrics.f1_score(y_true, y_pred)
    pass


def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):
    '''
    :param x_train: ciag treningowy wejsciowy NxM
    :param y_train: ciag treningowy wyjsciowy Nx1
    :param x_val: ciag walidacyjny wejsciowy Nval x M
    :param y_val: ciag walidacyjny wyjsciowy Nval x 1
    :param w0: wektor poczatkowych wartosci parametrow
    :param epochs: liczba epok dla SGD
    :param eta: krok uczenia
    :param mini_batch: wielkosc mini batcha
    :param lambdas: lista wartosci parametru regularyzacji lambda, ktore maja byc sprawdzone
    :param thetas: lista wartosci progow klasyfikacji theta, ktore maja byc sprawdzone
    :return: funckja wykonuje selekcje modelu. Zwraca krotke (regularization_lambda, theta, w, F), gdzie regularization_lambda
    to najlpszy parametr regularyzacji, theta to najlepszy prog klasyfikacji, a w to najlepszy wektor parametrow modelu.
    Dodatkowo funkcja zwraca macierz F, ktora zawiera wartosci miary F dla wszystkich par (lambda, theta). Do uczenia nalezy
    korzystac z algorytmu SGD oraz kryterium uczenia z regularyzacja l2.
    '''
    f_measure_best = -(2 ** 10)
    lambda_best = -1
    theta_best = -1
    w_best = np.zeros(w0.shape)
    f_measures = np.zeros((lambdas.shape[0], thetas.shape[0]))

    for lambda_i in range(lambdas.shape[0]):
        reg_logistic_partial = partial(regularized_logistic_cost_function, regularization_lambda=lambdas[lambda_i])
        w, _ = stochastic_gradient_descent(reg_logistic_partial, x_train, y_train, w0, epochs, eta, mini_batch)
        for theta_j in range(thetas.shape[0]):
            f_measure_i_j = f_measure(y_val, prediction(x_val, w, thetas[theta_j]))
            f_measures[lambda_i, theta_j] = f_measure_i_j
            if f_measure_i_j > f_measure_best:
                f_measure_best = f_measure_i_j
                lambda_best = lambdas[lambda_i]
                theta_best = thetas[theta_j]
                w_best = w
    return lambda_best, theta_best, w_best, f_measures
    pass
