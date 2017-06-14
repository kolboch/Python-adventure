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
import hog
import nn_utils as nn

TRAIN_DATA_FILE_PATH = 'train.pkl'
TRAIN_HOG_FILE_PATH = 'features_x_train.pkl'
WEIGHTS_HIDDEN_PATH = 'wh.pkl'
BIASES_HIDDEN_PATH = 'bh.pkl'
WEIGHTS_OUTPUT_PATH = 'wout.pkl'
BIASES_OUTPUT_PATH = 'bout.pkl'

NUMBER_OF_LABELS = 36

HOG_CELL_SIZE = 4
HOG_CELL_BLOCK = 3
HOG_NBINS = 9

NN_HIDDEN_NEURONS = 500
LEARNING_RATE = 0.2
epochs = 1000


def load_training_data():
    with open(TRAIN_DATA_FILE_PATH, 'rb') as f:
        return pkl.load(f)


def launch_learning(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 35} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """
    x_train, y_train = load_training_data()

    x_train = prepare_x(x_train)
    y_train = prepare_y(y_train)
    x = prepare_x(x)

    hog_for_shape = hog.hog(x_train[0], cell_size=(HOG_CELL_SIZE, HOG_CELL_SIZE),
                            cells_per_block=(HOG_CELL_BLOCK, HOG_CELL_BLOCK),
                            signed_orientation=False, nbins=HOG_NBINS, visualise=False,
                            normalise=True, flatten=True, same_size=True)

    with open(TRAIN_HOG_FILE_PATH, 'rb') as f:
        features_train = pkl.load(f)

    print('features_train after load:{}'.format(features_train))
    print('features_train after load shape:{}'.format(features_train.shape))

    if features_train.shape != (x_train.shape[0], hog_for_shape.shape[0]):
        features_train = np.empty(shape=(x_train.shape[0], hog_for_shape.shape[0]))
        print('Need to recompute features for training set')
        for i in range(x_train.shape[0]):
            features_train[i] = hog.hog(x_train[i], cell_size=(HOG_CELL_SIZE, HOG_CELL_SIZE),
                                        cells_per_block=(HOG_CELL_BLOCK, HOG_CELL_BLOCK),
                                        signed_orientation=False, nbins=HOG_NBINS, visualise=False,
                                        normalise=True, flatten=True, same_size=True)

        with open(TRAIN_HOG_FILE_PATH, 'wb') as pickle_file:
            pkl.dump(features_train, pickle_file)

    # those lines are neccesary in upload version, above code will disappear however
    # features_x = np.empty(shape=(x.shape[0], hog_for_shape.shape[0]))
    #         for i in range(x.shape[0]):
    #                 features_x[i] = hog.hog(x[i], cell_size=(HOG_CELL_SIZE, HOG_CELL_SIZE),
    #                             cells_per_block=(HOG_CELL_BLOCK, HOG_CELL_BLOCK),
    #                             signed_orientation=False, nbins=HOG_NBINS, visualise=False,
    #                             normalise=True, flatten=True, same_size=True)

    input_layer_neurons = features_train.shape[1]
    hidden_layer_neurons = NN_HIDDEN_NEURONS
    output_neurons = NUMBER_OF_LABELS
    needs_init = False
    try:
        with open(WEIGHTS_HIDDEN_PATH, 'rb') as f:
            weights_hidden = pkl.load(f)
        with open(BIASES_HIDDEN_PATH, 'rb') as f:
            biases_hidden = pkl.load(f)
        with open(WEIGHTS_OUTPUT_PATH, 'rb') as f:
            weights_output = pkl.load(f)
        with open(BIASES_OUTPUT_PATH, 'rb') as f:
            biases_output = pkl.load(f)
    except EOFError:
        needs_init = True

    if needs_init or weights_hidden.shape != (input_layer_neurons, hidden_layer_neurons):
        print('starting learning')

        # all connections from every feature to every node in hidden layer
        weights_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
        biases_hidden = np.random.uniform(size=(1, hidden_layer_neurons))

        # all connections from every hidden_neuron to output neuron
        weights_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
        biases_output = np.random.uniform(size=(1, output_neurons))

    for i in range(epochs):
        print('weights hidden:{} {} {}'.format(weights_hidden[0][1], weights_hidden[0][2], weights_hidden[0][3]))
        # if using batches it will go here
        hidden_ins_w = np.dot(features_train, weights_hidden)
        hidden_layer_input = hidden_ins_w + biases_hidden
        hidden_activations = nn.sigmoid(hidden_layer_input)

        output_hidden_ins_w = np.dot(hidden_activations, weights_output)
        output_layer_input = output_hidden_ins_w + biases_output
        output = nn.sigmoid(output_layer_input)

        # back propagation
        print('starting back propagation:{}'.format(i))
        error = calc_error(output, y_train)
        slope_output_layer = nn.sigmoid_derivative(output)
        slope_hidden_layer = nn.sigmoid_derivative(hidden_activations)

        delta_output = slope_output_layer * error

        error_hidden = delta_output.dot(weights_output.T)
        delta_hidden_layer = error_hidden * slope_hidden_layer

        weights_output += hidden_activations.T.dot(delta_output) * LEARNING_RATE
        biases_output += np.sum(delta_output, axis=0, keepdims=True) * LEARNING_RATE

        weights_hidden += features_train.T.dot(delta_hidden_layer) * LEARNING_RATE
        biases_hidden += np.sum(delta_hidden_layer, axis=0, keepdims=True) * LEARNING_RATE

        with open(WEIGHTS_HIDDEN_PATH, 'wb') as f:
            pkl.dump(weights_hidden, f)
        with open(BIASES_HIDDEN_PATH, 'wb') as f:
            pkl.dump(biases_hidden, f)
        with open(WEIGHTS_OUTPUT_PATH, 'wb') as f:
            pkl.dump(weights_output, f)
        with open(BIASES_OUTPUT_PATH, 'wb') as f:
            pkl.dump(biases_output, f)

    return 1
    pass


def calc_error(output, y_true):
    return y_true - output


def prepare_y(y_set):
    samples = y_set.shape[0]
    result = np.zeros(shape=(samples, NUMBER_OF_LABELS))
    for i in range(samples):
        result[i, y_set[i]] = 1
    return result


def prepare_x(x_to_prepare):
    N = x_to_prepare.shape[0]
    rc_cut = 4
    # x = np.reshape(x_to_prepare, (N, 56, 56))[:, rc_cut:-rc_cut, rc_cut:-rc_cut]
    # x = np.reshape(x[:, rc_cut:-rc_cut, rc_cut:-rc_cut], (N, (56 - rc_cut * 2) ** 2))
    # test v 2
    # x = x_to_prepare[:, :2500]
    return np.reshape(x_to_prepare, (N, 56, 56))[:, rc_cut:-rc_cut, rc_cut:-rc_cut]


def measure_error(y_predicted, y_true):
    error = 0
    print('predictions size:{}'.format(y_predicted.shape))
    print('y_true size:{}'.format(y_true.shape))
    for i in range(y_predicted.shape[0]):
        if y_predicted[i] != y_true[i]:
            error += 1
    return error / y_predicted.shape[0]


if __name__ == "__main__":
    x, y = load_training_data()
    launch_learning(x)
