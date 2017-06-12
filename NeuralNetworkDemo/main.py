# created by Karol 6-12-17
#
#
# Neural network demo, educational purposes without any application yet


import numpy as np

X = np.array([[1, 0, 1, 0],
              [1, 0, 1, 1],
              [0, 1, 0, 1]])

Y = np.array([[0], [1], [0]])

learning_rate = 0.1
epochs = 5000

input_layer_neurons = X.shape[1]  # depends from features we have in data
hidden_layer_neurons = 3  # we choose that number
output_neurons = 1  # in our case we have binary (0/1) output so one is enough


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def runner():
    # all connections from every feature to every node in hidden layer
    weights_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
    biases_hidden = np.random.uniform(size=(1, hidden_layer_neurons))

    # all connections from every hidden_neuron to output neuron
    weights_output = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
    biases_output = np.random.uniform(size=(1, output_neurons))

    for i in range(epochs):
        # forward propagation
        hidden_ins_w = np.dot(X, weights_hidden)  # NxM @ MxD
        hidden_layer_input = hidden_ins_w + biases_hidden  # NxD + 1xD, 1xD will be added to each row
        hidden_activations = sigmoid(hidden_layer_input)

        output_hidd_ins_w = np.dot(hidden_activations, weights_output)
        output_layer_input = output_hidd_ins_w + biases_output
        output = sigmoid(output_layer_input)

        # back propagation
        error = calculate_error(output, Y)
        print('error:{}'.format(error))
        slope_output_layer = sigmoid_derivative(output)
        slope_hidden_layer = sigmoid_derivative(hidden_activations)

        delta_output = slope_output_layer * error

        error_hidden = delta_output.dot(weights_output.T)
        delta_hidden = error_hidden * slope_hidden_layer

        weights_output += hidden_activations.T.dot(delta_output) * learning_rate
        biases_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate

        weights_hidden += X.T.dot(delta_hidden) * learning_rate
        biases_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

        print('output:{}'.format(output))
    return output
    pass


def calculate_error(y, y_true):
    return y_true - y
    # return ((y_true - y) ** 2) / 2


if __name__ == "__main__":
    runner()
