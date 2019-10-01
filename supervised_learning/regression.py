
import sys
dir_str = '../'
if (dir_str not in sys.path):
        sys.path.append(dir_str)

import numpy as np
from utils import normalize, polynomial_features
from utils import mean_squared_error

class Regularization():
    """
    Lasso Regression: l1:l2 = 1:0
    Ridge Regression: l1:l2 = 0:1
    Elastic Net Regression: l1:l2 = l1: (1-l1)
    """
    def __init__(self, alpha, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio=l1_ratio
    def __call__(self, w):
        #manhattan distance
        l1 = self.l1_ratio * np.sum(np.abs(w)) / w.shape[1]
        #euclidean distance
        l2 = ((1 - self.l1_ratio) * np.sum(w**2) / w.shape[1]) / 2.0
        return self.alpha * (l1 + l2)

    def grad(self, w):
        #derivative of absolute value function
        #https://socratic.org/questions/what-is-the-derivative-of-an-absolute-value
        dL_l1 = self.l1_ratio * np.sum(np.sign(w)) / w.shape[1]
        dL_l2 = (1 - self.l1_ratio) * np.sum(w) / w.shape[1]
        return self.alpha * (dL_l1 + dL_l2)

class BaseRegression(object):
    """
    Base model for regression

    Parameters
    -------------
    n_iterations: int
      number of iterations in training step the algorithm will tune the weights for
    learning_rate: float
      The step length the algorithm uses to update weights
    """

    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.train_loss = []
        self.val_loss = []
        self.params ={}
        self.grads = {}

    def fit(self, X, y, X_val=None, y_val=None):
        #for converting pandas or list -> ndarray
        self.X = np.array(X)
        self.y = np.array(y).reshape(-1, 1)
        self.X_val = np.array(X_val) if X_val is not None else None
        self.y_val = np.array(y_val).reshape(-1, 1) if y_val is not None else None

        self._initialize_weights(n_features=X.shape[1])
        self._train()

    def predict(self, X_test):
        X_test = np.array(X_test)
        return self._calc_pred(X_test)

    def _initialize_weights(self, n_features):
        limit = 1 / np.sqrt(n_features)
        #weight
        #generate rand from -limit to limit
        self.params['w'] = np.random.uniform(limit, limit, (n_features, 1))
        #bias
        self.params['b'] = 0

    def _train(self):
        for itr in range(self.n_iterations):
            #predict
            y_pred = self._calc_pred(self.X)

            #calculate loss
            mse = self._loss(self.y, y_pred)
            self.train_loss.append(mse)
            print("iter: %d, train_loss: %.3f" % (itr+1, mse), end="")
            #same calculation for validation data
            if not ((self.X_val is None) or (self.y_val is None)):
                y_pred_val = self._calc_pred(self.X_val)
                mse_val = self._loss(self.y_val, y_pred_val)
                self.val_loss.append(mse_val)
                print(", val_loss:  %.3f" % (mse_val), end="")

            print("\n", end="")
            #update weights
            self._update_coef(y_pred)

    def _calc_pred(self, X):
        return np.dot(X, self.params['w']) + self.params['b']

    def _loss(self, y_target, y_pred):
        #calculate loss
        #divided by two for derivative calculation
        return mean_squared_error(y_target, y_pred) + self.regularization(self.params['w'])

    def _update_coef(self, y_pred):
        #gradient of loss to weight(dL_dw, dL_db)
        self.grads['w'] = np.dot((y_pred - self.y).T, self.X)/len(self.y) + self.regularization.grad(self.params['w'])
        self.grads['b'] = np.sum(y_pred - self.y)/len(self.y)

        #update weights
        #here cannot be "w -= learning_rate * dw" but must be "w = w - learning_rate * dw"
        #https://stackoverflow.com/questions/47493559/valueerror-non-broadcastable-output-operand-with-shape-3-1-doesnt-match-the
        self.params['w'] = self.params['w'] - self.learning_rate * self.grads['w']
        self.params['b'] = self.params['b'] - self.learning_rate * self.grads['b']

class LinearRegression(BaseRegression):
    """
    Parameters
    -------------
    n_iterations: int
      number of iterations in training step the algorithm will tune the weights for
    learning_rate: float
      The step length the algorithm uses to update weights
    """
    def __init__(self, n_iterations=100, learning_rate=1e-3):
        #No regularization term for default
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(n_iterations=n_iterations, learning_rate=learning_rate)

class PolynomialRegression(BaseRegression):
    """
    Non-lenear transformation of the data before fitting the model

    Parameters
    ------------
    degree: int
      The defree of the polynominal that the independent variable X will be transformed to.
    n_iterations: int
      number of iterations in training step the algorithm will tune the weights for
    learning_rate: float
      The step length the algorithm uses to update weights
    """

    def __init__(self, degree=1, n_iterations=100, learning_rate=1e-3):
        super(PolynomialRegression, self).__init__(n_iterations=n_iterations, learning_rate=learning_rate)
        self.degree = degree
        #No regularization term for default
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0

    def fit(self, X, y, X_val=None, y_val=None):
        X = normalize(polynomial_features(X, degree=self.degree))
        X_val = None if X_val is None else normalize(polynomial_features(X_val, degree=self.degree))
        super(PolynomialRegression, self).fit(X=X, y=y, X_val=X_val, y_val=y_val)

    def predict(self, X_test):
        X_test = normalize(polynomial_features(X_test, degree=self.degree))
        super(PolynomialRegression, self).predict(X_test=X_test)

class LassoRegression(PolynomialRegression):
    def __init__(self, degree=0, reg_factor=0.05, n_iterations=100, learning_rate=1e-3):
        self.regularization = Regularization(alpha=reg_factor, l1_ratio=1.0)
        super(LassoRegression, self).__init__(degree=degree, n_iterations=n_iterations, learning_rate=learning_rate)


class RidgeRegression(PolynomialRegression):
    def __init__(self, degree=0, reg_factor=0.05, n_iterations=100, learning_rate=1e-3):
        self.regularization = Regularization(alpha=reg_factor,l1_ratio=0.0)
        super(RidgeRegression, self).__init__(degree=degree, n_iterations=n_iterations, learning_rate=learning_rate)

class ElasticNet(PolynomialRegression):
    def __init__(self, degree=0, reg_factor=0.05, l1_ratio=0.5, n_iterations=100, learning_rate=1e-3):
        self.regularization = Regularization(alpha=reg_factor,l1_ratio=l1_ratio)
        super(ElasticNet, self).__init__(degree=degree, n_iterations=n_iterations, learning_rate=learning_rate)


