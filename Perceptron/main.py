import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from perceptron import Perceptron
from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('o', 'x', 's', '^', 'v')
    colors = ('red', 'green', 'blue', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])  # unique labels

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


def main():
    data = pd.read_csv('data/iris.data', header=None)
    """
        columns in iris.data are as following
        sepal length | sepal width | petal length | petal width | species

        first we will show how our data looks like
    """
    # print('test how long it is:{}'.format(data.iloc[:, :].values.shape))
    y = data.iloc[0:100, 4].values  # labels of data ( species column), we take first 100 to work only on 2 classes
    y = np.where(y == 'Iris-setosa', -1, 1)  # now were Iris-setosa we set class to -1, else class is 1

    X = data.iloc[0:100, [0, 2]].values  # extracted sepal and petal length

    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='iris-setosa')
    plt.scatter(X[50:, 0], X[50:, 1], color='green', marker='x', label='iris-versicolor')
    plt.title('iris-setosa and iris-versicolor')
    plt.xlabel('sepal length[cm]')
    plt.ylabel('petal length[cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.draw()
    plt.waitforbuttonpress()

    my_prcpt = Perceptron(eta=0.1, n_iter=10)

    my_prcpt.fit(X, y)  # fitting with sepal and petal length (X) and classes (y)

    # now present how many errors where made when learning during each data traversal
    plt.figure()
    plt.plot(range(1, len(my_prcpt.errors_) + 1), my_prcpt.errors_, marker='.')
    plt.xticks(np.arange(1, len(my_prcpt.errors_) + 1, 1.0))
    plt.title('Errors made in following epochs')
    plt.xlabel('epochs')
    plt.ylabel('errors in prediction')
    plt.tight_layout()
    plt.draw()
    plt.waitforbuttonpress()

    plt.figure()
    plot_decision_regions(X, y, classifier=my_prcpt)
    plt.title('Decision regions')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.draw()
    plt.waitforbuttonpress()

if __name__ == '__main__':
    main()
