from itertools import combinations_with_replacement
import numpy as np
import sys

def polynomial_features(X, degree):
    X = np.array(X)
    n_samples, n_features = X.shape

    def index_combinations():
        combs = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs

    combinations = index_combinations()
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))

    for i, index_combs in enumerate(combinations):
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new

def normalize(X, axis=-1, order=2):
    """
    Normalize the dataset X
    """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    #return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    #insted of above considering the calculation cost
    return -2 + 1/(np.exp(2*x) + 1)

def relu(x):

    return x

def softmax(x):
    #return (np.exp(x))/sum_i((np.exp(x_i))
    e_x = np.exp(x).reshape(len(x), -1)
    return e_x / (np.sum(e_x, axis=1).reshape(-1, 1))
