import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

class SVM(object):
    """
    Support vector machine classifier

    Parameters
    ------------
    """

    def __init__(self, n_iterations=100, learning_rate=1e-5, threshold=1e-4, kernel='rbf', gamma=1, power=4, coef=4):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.threshold = threshold
        #self.C = C #regularization term
        self.train_margin = []
        self.val_margin = []
        self.params = {}
        self.grads = {}
        self.kernel = self._kernel(kernel, gamma=gamma, power=power, coef=coef) #function
        self.unique_value = None
        self.n_samples = None

    def fit(self, X, y, X_val=None, y_val=None):
        self.X = np.array(X)
        self.y = np.array(y).reshape(-1, 1)
        self.X_val = np.array(X_val) if X_val is not None else None
        self.y_val = np.array(y_val).reshape(-1, 1) if y_val is not None else None

        #(a, b) -> (0, 1) for y
        self.unique_value = np.unique(self.y)
        self.y = (self.y != self.unique_value[0]).astype(int)
        #(0, 1) -> (-1, 1) for y
        #(max - min)*y + min
        self.y = (1 - (-1) ) * self.y + (-1)

        #same manipulation for validation data
        if (self.y_val is not None):
            self.y_val = (self.y_val != self.unique_value[0]).astype(int)
            self.y_val = (1 - (-1) ) * self.y_val + (-1)

        self._initialize_weights(n_samples=X.shape[0], n_features=X.shape[1])
        self._train()


    def predict(self, X_test):
        X_test = np.array(X_test)
        return self._calc_pred(X_test)

    def _initialize_weights(self, n_samples, n_features):
        #lagrange coefficient
        self.n_samples = n_samples
        self.params['lambda'] = np.zeros((self.n_samples, 1))
        self.kernel_matrix = np.zeros((self.n_samples, self.n_samples))

        self.params['w'] = np.zeros((1, n_features))
        self.params['b'] = 0

    def _kernel(self, kernel, gamma, power, coef):
        if (kernel == 'linear'):
            def f(x1, x2):
                x1 = x1.reshape(1, -1)#(n_features,) -> (1, n_features)
                x2 = x2.reshape(1, -1)
                return np.dot(x1, x2.T)
            return f

        elif (kernel == 'rbf'):
            def f(x1, x2):
                x1 = x1.reshape(1, -1)#(n_features,) -> (1, n_features)
                x2 = x2.reshape(1, -1)
                return gamma * np.power(np.dot(x1, x2.T) + coef, power)
            return f

    def _train(self):
        for itr in range(self.n_iterations):
            y_pred = self._calc_pred(self.X)
            train_margin = self._calc_lagrangian(self.y, y_pred)
            self.train_margin.append(train_margin)
            print("iter: %d, train_margin: %.3f" % (itr+1, train_margin), end="")
            if not ((self.X_val is None) or (self.y_val is None)):
                y_pred_val = self._calc_pred(self.X_val)
                val_margin = self._calc_lagrangian(self.y_val, y_pred_val)
                self.val_margin.append(val_margin)
                print(", val_margin:  %.3f" % (val_margin), end="")
            print("\n", end="")
            self._update_coef()

    #too big values to be modified
    def _calc_lagrangian(self, y_target, y_pred):
        #https://scicomp.stackexchange.com/questions/2095/calculating-lagrange-coefficients-for-svm-in-python
        #lagrangian = np.sum(self.params['w']) - np.sum(np.dot(np.dot((y_target * self.params['w']).T, X), np.dot(X.T, y_target * self.params['w'])))/2.0
        #(((self.lmbd * y).T @ X) @ (X.T @ (self.lmbd * y)))/2
        lagrangian = np.mean(self.params['w']) - np.mean((y_target * y_pred - 1) @ self.params['w'])/2.0
        return lagrangian/len(y_target)

    def _calc_pred(self, X):
         y_pred = np.dot(X, self.params['w'].T) + self.params['b']
         y_pred = (y_pred > 0).astype(int)
         return y_pred

    def _update_coef(self):
        for s in range(self.n_samples):
            for t in range(self.n_samples):
                self.kernel_matrix[s, t] = self.params['lambda'][t]*self.y[s]*self.y[t]*self.kernel(self.X[s, :], self.X[t, :]) + self.params['lambda'][s]*self.y[s]*self.y[t]
        self.grads['lambda'] = (1 - np.sum(self.kernel_matrix, axis=1,keepdims=True))/self.n_samples
        self.params['lambda'] = self.params['lambda'] - self.learning_rate * self.grads['lambda']
        #index of support vector
        self.sv_idx = (self.params['lambda'] > self.threshold).reshape(-1,)
        self.params['w'] = np.dot(((self.params['lambda'] * self.y)[self.sv_idx]).T, self.X[self.sv_idx])/(self.n_samples)
        self.params['b'] = np.mean(self.y[self.sv_idx] - np.dot(self.X[self.sv_idx], self.params['w'].T)/(self.n_samples))

    def decision_region(self, X, y, step=0.01, title='decision region', xlabel='xlabel', ylabel='ylabel',  target_names=['1', '-1']):
        """
        plot decision region for the learned model learned with 2 features with two labels

        background color represents predicted class
        plot color shows actual target value

        Parameters
        ---------------
        X: ndarray, shape(n_samples, 2)
            features
        y: ndarray, shape(n_samples,)
            target
        model: object
            learned model instance
        step: float, (default: 0.1)
            step to calculate estimation
        title: str
            graph title
        xlabel, ylabel: str
            axis label
        target_names=: list of str
            legend name
        ---------------
        """
        #setting
        scatter_color = ['red', 'blue']
        contourf_color = ['skyblue', 'pink']
        n_class = 2


        #pred
        #gen mesh for each features（when a=(a1, a2, a3), b=(b1, b2, b3), generate [[a1, b1], [a1, b2], [a1, b3]], [[a2, b1], [a2, b2], [a2, b3]], [[a3, b1], [a3, b2], [a3, b3]]）
        mesh_f0, mesh_f1 = np.meshgrid(np.arange(np.min(X[:,0])-0.5, np.max(X[:,0])+0.5, step), np.arange(np.min(X[:,1])-0.5, np.max(X[:,1])+0.5, step))

        #multi-dim to one-dim
        mesh = np.c_[np.ravel(mesh_f0),np.ravel(mesh_f1)]
        pred = self.predict(mesh).reshape(mesh_f0.shape)

        #plot
        fig, ax = plt.subplots(1,1, figsize=(6,4 ))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.contourf(mesh_f0, mesh_f1, pred, n_class-1, cmap=ListedColormap(contourf_color))

        ax.contour(mesh_f0, mesh_f1, pred, n_class-1, colors='y', linewidths=3, alpha = 0.5)
        #plot for un-overlapping element
        for i, target in enumerate(set(np.unique(y))):
            ax.scatter(X[y==target][:, 0], X[y==target][:, 1], s=20, color=scatter_color[i], label=target_names[i], marker='o')

        #when plotting for train data
        if (self._check_same_ary(y, self.y)):
            #yellow color for support vectors
            ax.scatter(X[self.sv_idx, 0], X[self.sv_idx, 1], s=20, color='yellow' , label='supprt vector', marker='o')

        ax.legend()


    def _check_same_ary(self, a, b):
        return (a == b).all() if (a.shape == b.shape) else False

    #plot learning curve
    def plot_learning_curve(self):
        fig, ax = plt.subplots(1,1, figsize=(4,4 ))
        ax.plot(np.array(range(self.n_iterations)), self.train_margin, "-", label = 'train')
        ax.plot(np.array(range(self.n_iterations)), self.val_margin, "-", label = 'validation')

        #label
        ax.set_title('Learning Curve')
        ax.set_xlabel('n of iterations')
        ax.set_ylabel('Lagrangian')
        ax.legend()
