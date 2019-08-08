import numpy as np
import math

def mean_squared_error(y_target, y_pred):
    return np.mean(np.power(y_pred - y_target, 2))

def cross_entropy_error(y_target, y_pred):
    return - np.mean(y_target*(np.log(y_pred)) + (1 - y_target) * (np.log(1- y_pred)))

def calculate_entropy(y):
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy

def calculate_variance(X):
    X = np.array(X)
    n_samples, n_features = X.shape
    mean = np.ones((n_samples, n_features)) * X.mean(0)
    variance = (1/n_samples) * np.diag(np.dot((X - mean).T, X-mean))
    return variance

def divide_on_feature(X, feature_idx, threshold):
    """
    Divide dataset based on if sample value on feature index is larger than the given threshold
    """

    if (isinstance(threshold, int) or (isinstance(threshold, float))):
        split_func = lambda sample: sample[feature_idx] >= threshold
    else:
        split_func = lambda sample: sample[feature_idx] == threshold

    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return np.array([X_1, X_2])


def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred, axis=0)/len(y_true)

def calculate_covariance_matrix(X, y=None):
    if (y is None):
        y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples - 1)) * np.dot((X - X.mean(axis=0)).T, y - y.mean(axis=0))
    return np.array(covariance_matrix, dtype=float)

def get_random_subsets(X, y, n_subsets, replacements=True):
    """
    Returns random subsets (with replacement) of the data
    replacements True means sampling from pool with redundancy
    """
    n_samples = np.shape(X)[0]
    #concatenate x and y
    X_y = np.concatenate((X, y.reshape(-1, 1)), axis=1)
    #randomly shuffle
    np.random.shuffle(X_y)
    subsets = []

    #Uses 50% of training samples without replacement
    subsample_size = int(n_samples / 2)
    if replacements:
        subsample_size = n_samples

    for _ in range(n_subsets):
        idx = np.random.choice(
            range(n_samples),
            size=np.shape(range(subsample_size)),
            replace=replacements
        )
        X = X_y[idx][:, :-1]
        y = X_y[idx][:, -1]
        subsets.append([X, y])

    return subsets

