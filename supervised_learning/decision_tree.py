import numpy as np
from utils import divide_on_feature, calculate_variance, calculate_entropy

class DecisionNode():
    """
    Node or leaf in the decision tree
    """

    def __init__(self, feature_idx=None, threshold=None, value=None, true_branch=None, false_branch=None):
        self.feature_idx = feature_idx #index for the feature that is tested
        self.threshold = threshold #threshold value for feature
        self.value = value #value if the node is a leaf in the tree
        self.true_branch = true_branch #left subtree
        self.false_branch = false_branch #right subtree


class DecisionTree(object):
    """
    Super class of regression tree and classification tree
    """

    def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=float('inf'), loss=None):
        self.root = None #root node in decision tree
        self.min_samples_split = min_samples_split #min n of samples to justify split
        self.min_impurity = min_impurity #min impurity to justify split
        self.max_depth = max_depth #max depth to grow the tree to
        self._impurity_calculation = None #function to calculate impurity( info gain for classificaiton, variance reduction for regression)
        self._leaf_value_calculation = None #function to determine prediction y at leaf

        self.one_dim = None #if y is one-hot encoded (multi-dim) or not (one-dim)
        self.loss = loss #if gradient boost
        self.feature_indices=None #for random forest


    def fit(self, X, y, loss=None):
        """
        Build decision tree
        """
        self.one_dim = (len(np.shape(y)) == 1)
        self.root = self._build_tree(X, y)
        self.loss = None

    def _build_tree(self, X, y, current_depth=0):
        """
        Recursive method which builds out the decision tree and splits X and respective y on the feature of X which (based on impurity) best separates the data
        """
        X = np.array(X)
        y = np.array(y).reshape(len(y), -1)

        largest_impurity = 0
        best_criteria = None #feature index and threshold
        best_sets = None #subsets of the data

        n_samples, n_features = np.shape(X)

        if (n_samples >= self.min_samples_split and current_depth <= self.max_depth):
            #calculate the impurity for each feature
            for feature_idx in range(n_features):
                #all values of feature_idx
                feature_values = X[:, feature_idx].reshape(-1, 1)
                unique_values = np.unique(feature_values)

                #iterate through all unique values of feature column i and calculate the impurity
                for threshold in unique_values:
                    #devide X and y depending on if the feature value of X at index feature_idx meets the threshold
                    idx_1, idx_2 = divide_on_feature(X, feature_idx, threshold)

                    X1 = X[idx_1, :]
                    X2 = X[idx_2, :]
                    y1 = y[idx_1, :]
                    y2 = y[idx_2, :]

                    if (len(X1) > 0 and len(X2) > 0):
                        #calculate impurity
                        impurity = self._impurity_calculation(y, y1, y2)

                        #save the threshold value and the feature index if this threshold resulted in a higher informaiton gain than previously recorded
                        if (impurity > largest_impurity):
                            largest_impurity = impurity
                            best_criteria = {"feature_idx": feature_idx, "threshold": threshold}
                            best_sets = {"leftX": X1, "lefty": y1, "rightX": X2, "righty": y2}
        if (largest_impurity > self.min_impurity):
            #build subtree for the right and left branches
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth+1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth+1)
            return DecisionNode(feature_idx=best_criteria["feature_idx"], threshold=best_criteria["threshold"], true_branch=true_branch, false_branch=false_branch)

        #determine value if it reaches at a leaf
        leaf_value = self._leaf_value_calculation(y)
        return DecisionNode(value=leaf_value)

    def predict_value(self, X, tree=None):
        """
        Do a recursive serch down the tree and make a prediction of the data sample by the value of the leaf that we end up at
        """
        if (tree is None):
            tree = self.root

        #return value as the prediction if we have a value (i.e. we are at a leaf)
        if (tree.value is not None):
            return tree.value

        #choose the feature that we will test
        feature_value = X[tree.feature_idx]
        #determine if we will follow left or right branch
        branch = tree.false_branch
        if (isinstance(feature_value, int) or isinstance(feature_value, float)):
            if (feature_value >= tree.threshold):
                branch = tree.true_branch
        elif (feature_value == tree.threshold):
            branch = tree.true_branch


        #test subtree
        return self.predict_value(X, branch)

    def predict(self, X):
        """
        """
        y_pred = [self.predict_value(sample) for sample in X]
        return np.array(y_pred)

    def print_tree(self, tree=None, indent=" "):
        """
        Recursively print the decision tree
        """
        if (tree is None):
            tree = self.root

        #print the label if we are at leaf
        if (tree.value is not None):
            print(tree.value)
        #Go deeper down the tree
        else:
            #print test
            print("%s:%s? " % (tree.feature_idx, tree.threshold))
            #print the true scenario
            print("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)

            #print the false scenario
            print("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)

class XGBoostRegressionTree(DecisionTree):
    """
    Regression tree for XGBoost
    """

    def _split(self, y):
        """
        y contains y_true in left half of the middle column and y_pred in the right half. Split and return the two matrices
        """

        col = int(np.shape(y)[1]/2)
        y, y_pred = y[:, :col], y[:, col:]
        return y, y_pred

    def _gain(self, y, y_pred):
        nominator = np.power(np.sum(y * self.loss.gradient(y, y_pred)), 2)
        denominator = np.sum(self.loss.hess(y, y_pred))
        return 0.5 * (nominator / denominator)

    def _gain_by_taylor(self, y, y1, y2):
        #split
        y, y_pred = self._split(y)
        y1, y1_pred = self._split(y1)
        y2, y2_pred = self._split(y2)

        true_gain = self._gain(y1, y1_pred)
        false_gain = self._gain(y2, y2_pred)
        gain = self._gain(y, y_pred)
        return true_gain + false_gain - gain

    def _apprpximate_update(self, y):
        #y split into y, y_pred
        y, y_pred = self._split(y)
        #Newton's Method
        gradient = np.sum(y*self.loss.gradient(y, y_pred), axis=0)
        hessian = np.sum(self.loss.hess(y, y_pred), axis=0)
        update_approximation = gradient / hessian

        return update_approximation

    def fit(self, X, y):
        self._impurity_calculation = self._gain_by_taylor
        self._leaf_value_calculation = self._apprpximate_update
        super(XGBoostRegressionTree, self).fit(X, y)


class RegressionTree(DecisionTree):
    def _calculate_variance_reduction(self, y, y1, y2):
        var_tot = calculate_variance(y)
        var_1 = calculate_variance(y1)
        var_2 = calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)

        #calculate the variance reduction
        variance_reduction = var_tot - (frac_1*var_1 + frac_2*var_2)

        return sum(variance_reduction)

    def _mean_of_y(self, y):
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree, self).fit(X, y)

class ClassificationTree(DecisionTree):
    def _calculate_information_gain(self, y, y1, y2):
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        information_gain = entropy - (p * calculate_entropy(y1) + (1-p) * calculate_entropy(y2))
        return information_gain

    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            #count number of occurences of samples with label
            count = len(y[y==label])
            if (count > max_count):
                most_common = label
                max_count = count

        return most_common

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)
