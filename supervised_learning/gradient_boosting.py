import sys

dir_str = ""
if (dir_str not in sys.path):
	sys.path.append(dir_str)

import numpy as np
import progressbar

from deep_learning.loss_functions import SquareLoss, CrossEntropy
from utils import bar_widgets
from utils import softmax, to_categorical
from supervised_learning import RegressionTree, ClassificationTree

class GradientBoosting(object):
	"""
	"""

	def __init__(self, n_estimators, learning_rate, min_samples_split, min_impurity, max_depth, regression):
		self.n_estimators = n_estimators
		self.learning_rate = learning_rate
		self.min_samples_split = min_samples_split
		self.min_impurity = min_impurity
		self.max_depth = max_depth
		self.regression = regression
		self.bar = progressbar.ProgressBar(widgets=bar_widgets)
		self.n_classes = None

		if self.regression:
			self.train_loss = SquareLoss()
		else:
			self.train_loss = CrossEntropy()

		self.trees = []
		for _ in range(self.n_estimators):
			
			tree = RegressionTree(
					min_samples_split=self.min_samples_split,
					min_impurity = self.min_impurity,
					max_depth = self.max_depth
				)

			self.trees.append(tree)

	def fit(self, X, y):
		X = np.array(X)
		y = y.reshape(len(y), -1)
		y_pred = np.full(y.shape, np.mean(y, axis=0)) 
		for tree in self.bar(self.trees):
			grad = self.train_loss.backward(y, y_pred)
			tree.fit(X, grad)
			update = tree.predict(X)
			y_pred = y_pred - self.learning_rate * update.reshape(len(update), -1)
	def predict(self, X_test):
		X_test = np.array(X_test)
		y_pred = np.array([])
		for tree in self.trees:
			update = tree.predict(X_test)
			update = self.learning_rate * update.reshape(len(update), -1)
			y_pred = -update if not y_pred.any() else y_pred - update
		if not self.regression:
			y_pred = softmax(y_pred)
			y_pred = np.argmax(y_pred, axis=1)
		return y_pred

class GradientBoostingRegressor(GradientBoosting):
	def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2, min_var_red=1e-7, max_depth=4):
		super(GradientBoostingRegressor, self).__init__(
				n_estimators=n_estimators,
				learning_rate=learning_rate,
				min_samples_split=min_samples_split,
				min_impurity=min_var_red,#variance reduction
				max_depth=max_depth,
				regression=True
			)

class GradientBoostingClassifier(GradientBoosting):
	def __init__(self, n_estimators=200, learning_rate=0.1, min_samples_split=2, min_info_gain=1e-7, max_depth=4):
		super(GradientBoostingClassifier, self).__init__(
				n_estimators=n_estimators,
				learning_rate=learning_rate,
				min_samples_split=min_samples_split,
				min_impurity=min_info_gain,#information_gain
				max_depth=max_depth,
				regression=False
			)

	def fit(self, X, y):
		y = to_categorical(y)
		super(GradientBoostingClassifier, self).fit(X, y)

