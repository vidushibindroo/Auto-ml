import numpy as np
import math
import sys
from Algorithms.XGBoost.DTreeXGB import decision_tree
from Algorithms.XGBoost.utils import mean_squared_error, accuracy_score, divide
from Algorithms.DecisionTree.utils import calculate_entropy, calculate_variance, standardize

class classification_tree(decision_tree):
    def information_gain(self, y, y1, y2):
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * \
            calculate_entropy(y1) - (1 - p) * \
            calculate_entropy(y2)

        return info_gain

    def majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X, y):
        self._impurity_calculation = self.information_gain
        self._leaf_value_calculation = self.majority_vote
        super(classification_tree, self).fit(X, y)