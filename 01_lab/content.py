# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  content.py zaimplementował K. Bochyński
#  2017
# --------------------------------------------------------------------------

import numpy as np
from utils import polynomial

#done
def mean_squared_error(x, y, w):
    '''
            :param x: ciag wejsciowy Nx1
            :param y: ciag wyjsciowy Nx1
            :param w: parametry modelu (M+1)x1
            :return: blad sredniokwadratowy pomiedzy wyjsciami y
            oraz wyjsciami uzyskanymi z wielomianu o parametrach w dla wejsc x
    '''

    #print('mean_squared_error x:\n{}\n'.format(x))
    #print('\nmean_squared_error y:\n{}\n'.format(y))
    #print('\nmean_squared_error w:\n{}\nw.length: {}'.format(w, w.shape[0]))

    poly_results = np.empty(x.shape)
    for i in range(x.shape[0]):
        power_counter = 0
        result = 0
        for param in w:
            result += np.power(x[i], power_counter) * param
            power_counter += 1
        poly_results[i] = result

    diff_results = y - poly_results
    sum_squared = 0
    for elem in diff_results:
        sum_squared += elem * elem

    return sum_squared / diff_results.shape[0]
    pass

#done :)
def design_matrix(x_train,M):
    '''
    :param x_train: ciag treningowy Nx1
    :param M: stopien wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzedu M
    '''
    #print('design_matrix x_train:\n{}\n'.format(x_train))
    #print('design_matrix M:\n{}\n'.format(M))
    generated_matrix = np.vander(x_train.flatten('F'), M + 1, True)
    #print('design_matrix design_matrix:\n{}\n'.format(generated_matrix))
    return generated_matrix
    pass


def least_squares(x_train, y_train, M):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu, a err blad sredniokwadratowy
    dopasowania
    '''
    (coefficients, residuals, rank, s) = np.linalg.lstsq(design_matrix(x_train, M), y_train)
    err = mean_squared_error(x_train, y_train, coefficients)
    return coefficients, err[0]
    pass


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu zgodnie z kryterium z regularyzacja l2,
    a err blad sredniokwadratowy dopasowania
    '''
    d_matrix = design_matrix(x_train, M)
    equation_part1 = np.linalg.inv(d_matrix.T.dot(d_matrix) + regularization_lambda * np.identity(d_matrix.shape[1]))
    w = equation_part1.dot(d_matrix.T).dot(y_train)
    err = mean_squared_error(x_train, y_train, w)
    return w, err[0]
    pass


def model_selection(x_train, y_train, x_val, y_val, M_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, ktore maja byc sprawdzone
    :return: funkcja zwraca krotke (w,train_err,val_err), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym, train_err i val_err to bledy na sredniokwadratowe na ciagach treningowym
    i walidacyjnym
    '''
    min_train_error = None
    min_validate_error = None
    min_coefficients = None
    for i in range(len(M_values)):
        (coefficients, train_error) = least_squares(x_train, y_train, M_values[i])
        current_valid_error = mean_squared_error(x_val, y_val, coefficients)
        if min_validate_error is None or current_valid_error < min_validate_error:
            min_validate_error = current_valid_error
            min_coefficients = coefficients
            min_train_error = train_error
    return min_coefficients, min_train_error, min_validate_error[0]
    pass


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M: stopien wielomianu
    :param lambda_values: lista ze wartosciami roznych parametrow regularyzacji
    :return: funkcja zwraca krotke (w,train_err,val_err,regularization_lambda), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym. Wielomian dopasowany jest wg kryterium z regularyzacja. train_err i val_err to
    bledy na sredniokwadratowe na ciagach treningowym i walidacyjnym. regularization_lambda to najlepsza wartosc parametru regularyzacji
    '''
    min_train_error = None
    min_validate_error = None
    min_lambda_value = None
    for i in range(len(lambda_values)):
        (coefficients, train_error) = regularized_least_squares(x_train, y_train, M, lambda_values[i])
        current_valid_error = mean_squared_error(x_val, y_val, coefficients)
        if min_validate_error is None or current_valid_error < min_validate_error:
            min_validate_error = current_valid_error
            min_coefficients = coefficients
            min_train_error = train_error
            min_lambda_value = lambda_values[i]
    return min_coefficients, min_train_error, min_validate_error[0], min_lambda_value
    pass