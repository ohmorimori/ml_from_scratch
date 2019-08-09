import numpy as np
import math


#Decision stump (one node decision tree) used as week classifier
class DecisionStump():
    def __init__(self):
        #Determines if sample shall be classified  as -1 or 1 given threshold
        self.polarity = 1
        #The index of the feature used to make classification
        self.feature_idx = None
        #The threshold value that the feature should be measured against
        self.threshold = None
        #Value indicative of the classifier' accuracy
        self.alpha = None

class Adaboost():
    """
    Boosting method that uses a number of weak classifier in ensemble to make a strong classifier using decision stump, which is one level decision tree.

    Parameters:
    --------------
    n_clf: int
      n of weak classifiers
    """
    def __init__(self, n_clf):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        #initialize weight as 1/N
        w = np.full(n_samples, (1/n_samples))
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')

            for feature_idx in range(n_features):
                feature_values = X[:, feature_idx]
                unique_values = np.unique(feature_values)
                #Try every unique feature value as threshold
                for threshold in unique_values:
                    polarity = 1
                    y_pred = np.ones_like(y)
                    #label the samples whose values are below threshold as '-1'
                    under_threshold_idx = feature_values < threshold

                    y_pred[under_threshold_idx] = -1
                    #wum of weights of misclassified samples
                    error = sum(w[(y != y_pred)])
                    #flip error ratio so that it will be 0.5 or more
                    if (error > 0.5):
                        error = 1 - error
                        polarity = -1

                    if (error < min_error):
                        clf.polarity = polarity
                        clf.threshold = threshold
                        clf.feature_idx = feature_idx
                        min_error = error
            #calculate the alpha which is used to update the sample weights.
            clf.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
            y_preds = np.ones(n_samples)
            negative_idx = (clf.polarity * X[:, clf.feature_idx] < clf.polarity * clf.threshold)
            y_preds[negative_idx] = -1
            w *= np.exp(-clf.alpha * y * y_preds)
            #normalize total to one
            w /= np.sum(w)

            self.clfs.append(clf)

    def predict(self, X_test):
        X_test = np.array(X_test)
        n_samples = X_test.shape[0]
        y_pred = np.zeros((n_samples, 1))

        for clf in self.clfs:
            predictions = np.ones_like(y_pred)
            negative_idx = (clf.polarity * X_test[:, clf.feature_idx] < clf.polarity * clf.threshold)
            predictions[negative_idx] = -1
            y_pred += clf.alpha * predictions
        y_pred = np.sign(y_pred).flatten()
        return y_pred
