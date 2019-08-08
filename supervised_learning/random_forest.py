import sys
dir_str = ".."
if (dir_str not in sys.path):
    sys.path.append(dir_str)
import numpy as np
import math
import progressbar

from utils import get_random_subsets, bar_widgets
from supervised_learning import ClassificationTree


class RandomForest():
    """
    Random Rofest classifer
    """
    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2, min_gain=0, max_depth=float("inf")):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_depth = max_depth
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

        #initialize decision trees
        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(
                ClassificationTree(
                    min_samples_split=self.min_samples_split,
                    min_impurity=self.min_gain,
                    max_depth=self.max_depth
                  )
              )
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_features = np.shape(X)[1]

        #select max_features if not defined
        if (self.max_features is None):
            self.max_features = int(math.sqrt(n_features))

        subsets = get_random_subsets(X, y, self.n_estimators)


        for i in self.progressbar(range(self.n_estimators)):
            X_subset, y_subset = subsets[i]
            #feature bagging
            idx = np.random.choice(range(n_features), size=self.max_features, replace=True)
            #save the indices of the features for prediction
            self.trees[i].feature_indices = idx
            X_subset = X_subset[:, idx]
            #fit the tree to the data
            self.trees[i].fit(X_subset, y_subset)

    def predict(self, X_test):
        y_preds = np.empty((X_test.shape[0], len(self.trees)))

        #let each trees predict on the data
        for i, tree in enumerate(self.trees):
            #Indices of the features that the tree has trained on
            idx = tree.feature_indices
            #make a prediction based on those features
            prediction = tree.predict(X_test[:, idx])
            y_preds[:, i] = prediction
        #not y_pred"s"
        y_pred = []
        #select the most common class prediction for each sample
        for sample_predictions in y_preds:
            y_pred.append(np.bincount(sample_predictions.astype('int')).argmax())
        return y_pred
