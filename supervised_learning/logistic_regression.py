import sys
dir_str = ".."
if (dir_str not in sys.path):
    sys.path.append(dir_str)

import numpy as np
#from utils import make_diagonal, Plot
from utils import sigmoid
from utils import cross_entropy_error
from .regression import Regularization

class LogisticRegression():
    """
    Logistic Regression Classifier
    Parameters
    -------------
    n_iterations: int
    learning_rate: float
    alpha: float
      coefficient for regularization term
    l1_ratio: float (0-1)
      the ratio that represents L1/(L1 + L2) for regularization term
    -------------
    """
    def __init__(self, n_iterations=100, learning_rate=1e-3, alpha=0, l1_ratio=0):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.train_loss = []
        self.val_loss = []
        self.params = {}
        self.grads = {}
        self.regularization = Regularization(alpha=alpha, l1_ratio=l1_ratio)

    def fit(self, X, y, X_val=None, y_val=None):
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
        limit = 1 / (np.sqrt(n_features))

        self.params['w'] = np.random.uniform(-limit, limit, (1, n_features))
        self.params['b'] = 0

    def _train(self):
        for itr in range(self.n_iterations):
            #predict
            y_pred = self._calc_pred(self.X)
            #calculate loss
            cross_entropy = self._calc_loss(self.y, y_pred)
            self.train_loss.append(cross_entropy)
            print("iter: %d, train_loss: %.3f" % (itr+1, cross_entropy), end="")

            #same calculation for validation data
            if not ((self.X_val is None) or (self.y_val is None)):
                y_pred_val = self._calc_pred(self.X_val)
                cross_entropy_val = self._calc_loss(self.y_val, y_pred_val)
                self.val_loss.append(cross_entropy_val)
                print(", val_loss:  %.3f" % (cross_entropy_val), end="")
            print("\n", end="")

            #update weights
            self._update_coef(y_pred)

    def _calc_pred(self, X):
        #calculate prediction
        return sigmoid(np.dot(X, self.params['w']) + self.params['b'])

    def _calc_loss(self, y_target, y_pred):
        #calculate loss
        return cross_entropy_error(y_target, y_pred) + self.regularization(self.params['w'])

    def _update_coef(self, y_pred):
        #gradient of loss to weight(dL/dw, dL/db)
        #dL/dw = dL/dy * dy/df * df/dw = (y - target)X
        #dL/dy = -{target * (1/y) + (1-target) *(-1)*(1/(1-y))}
        #dy/df = y(1-y), when y = sigmoid(f)
        #df/dw = X, since f = X * weight + bias
        #same for bias term but for df/db = 1
        self.grads['w'] = np.dot((y_pred - self.y).T, self.X)/len(self.y) + self.regularization.grad(self.params['w'])
        self.grads['b'] = np.sum((y_pred - self.y))/len(self.y)

        self.params['w'] = self.params['w'] - self.learning_rate * self.grads['w']
        self.params['b'] = self.params['b'] - self.learning_rate * self.grads['b']
