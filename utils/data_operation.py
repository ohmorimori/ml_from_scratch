import numpy as np


def mean_squared_error(y_target, y_pred):
    return np.mean(np.power(y_pred - y_target, 2))

def cross_entropy_error(y_target, y_pred):
    return - np.mean(y_target*(np.log(y_pred)) + (1 - y_target) * (np.log(1- y_pred)))
