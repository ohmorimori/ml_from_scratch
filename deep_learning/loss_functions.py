import sys
dir_str = ".."
if (dir_str not in sys.path):
	sys.path.append(dir_str)

import numpy as np
from utils import mean_squared_error, cross_entropy_error

class Loss(object):
	def __init__(self):
		pass

	def forward(self, y_true, y_pred):
		raise NotImplementedError()

	def backward(self, y, y_pred):
		raise NotImplementedError()

	def acc(self, y, y_pred):
		return 0

class SquareLoss(Loss):
	def forward(self, y, y_pred):
		return mean_squared_error(y, y_pred)

	def backward(self, y, y_pred):
		return - (y - y_pred)

class CrossEntropy(Loss):
	def forward(self, y, y_pred):
		return cross_entropy_error(y, y_pred)

	def backward(self, y, y_pred):
		# Avoid division by zero
		y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
		return - (y / y_pred) + (1 - y) / (1 - y_pred)
