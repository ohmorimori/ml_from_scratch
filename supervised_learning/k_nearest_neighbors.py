import sys
import numpy as np

dir_str = ".."
if dir_str not in sys.path:
	sys.path.append(dir_str)

from utils import euclidean_distance


class Knn():
	"""
	K Nearest neighbors classifier

	Parameters:
	-----------
	k: int
		The number of closest neighbors that will determine the class of the samples
	"""

	def __init__(self, k=5):
		self.k = k

	def _vote(self, neighbor_labels):
		"""
		Return the most common class among the neighbor samples
		"""
		counts = np.bincount(neighbor_labels.astype(np.int64))
		return counts.argmax()

	def predict(self, X_test, X_train, y_train):
		X_test = np.array(X_test)
		X_train = np.array(X_train)
		y_train = np.array(y_train)

		#Determine the class of each sample
		y_pred = np.empty(X_test.shape[0])
		for i, test_sample in enumerate(X_test):
			#sort the training samples by their distance to the test sample and get the K nearest
			idx = np.argsort([euclidean_distance(test_sample, x) for x in X_train])[:self.k]
			#Extract the labels of the K nearest neighboring training samples
			k_nearest_neighbors = np.array([y_train[i] for i in idx])
			#Label sample as the most common class label
			y_pred[i] = self._vote(k_nearest_neighbors)

		return y_pred
