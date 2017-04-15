import numpy as np


class Perceptron(object):
    """
    Perceptron classifier.
    -------------Parameters-----------
    eta: float
    Learning rate (between 0.0 and 1.0)
    n_iter: int
    Passes over the training dataset.
    -------------Attributes-----------
    w_: 1d-array
    weights after fitting
    errors_: list
    number of misclassifications in every approach
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Fit training data.
        :param X: {array-like}, shape = [n_samples, n_features], Training vectors
        where n_samples is the number of samples and n_features is the number of features
        :param y: array-like, shape = [n_samples], Target values
        :return: self: object
        """
        self.weigths_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.weigths_[1:] += update * xi
                self.weigths_[0] += update
                errors += int(update != 0.0)  # casting boolean value, if true we add 1 else 0
            self.errors_.append(errors)  # after single iteration is done remember how many errors in prediction we made
        return self

    def net_input(self, X):
        """calculate net input"""
        return np.dot(X, self.weigths_[1:]) + self.weigths_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
