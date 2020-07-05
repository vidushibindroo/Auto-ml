import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import progressbar



eps = 1e-8


class XGBoostRegressionTree(DecisionTree):
    
    '''Call from class decision tree, after importing it from decision_tree_functions in the automl project 
    as DecisionTree because we're using a decision tree ensemble (The decision tree function needs to be modified
    to a class for this to work). Refer: https://xgboost.readthedocs.io/en/latest/tutorials/model.html'''
  
    def _split(self, y):
        col = int(np.shape(y)[1]/2)
        y, y_pred = y[:, :col], y[:, col:]
        return y, y_pred

    def _gain(self, y, y_pred):
        nominator = np.power((y * self.loss.gradient(y, y_pred)).sum(), 2)
        denominator = self.loss.hess(y, y_pred).sum()
        return 0.5 * (nominator / denominator)

    def _gain_by_taylor(self, y, y1, y2):
        y, y_pred = self._split(y)
        y1, y1_pred = self._split(y1)
        y2, y2_pred = self._split(y2)

        true_gain = self._gain(y1, y1_pred)
        false_gain = self._gain(y2, y2_pred)
        gain = self._gain(y, y_pred)
        return true_gain + false_gain - gain

    def _approximate_update(self, y):
        y, y_pred = self._split(y)
        gradient = np.sum(y * self.loss.gradient(y, y_pred), axis=0)
        hessian = np.sum(self.loss.hess(y, y_pred), axis=0)
        update_approximation =  gradient / hessian

        return update_approximation

    def fit(self, X, y):
        self._impurity_calculation = self._gain_by_taylor
        self._leaf_value_calculation = self._approximate_update
        super(XGBoostRegressionTree, self).fit(X, y)


class log_loss():
    def __init__(self):
        sigmoid = utils.sigmoid_func()
        self.log_func = sigmoid
        self.log_grad = sigmoid.gradient

    def loss(self, y, y_pred):
        y_pred = np.clip(y_pred, eps, 1 - eps)
        p = self.log_func(y_pred)
        return y * np.log(p) + (1 - y) * np.log(1 - p)

    def gradient(self, y, y_pred):
        p = self.log_func(y_pred)
        return -(y - p)

    def hess(self, y, y_pred):
        p = self.log_func(y_pred)
        return p * (1 - p)


xgb_dic = {
    'ml_task':["classification"],
    'counter': [0],
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'min_samples_split': [1, 2, 3],
    'max_depth': [2, 3, 4, 5, 6, 7],
    'min_impurity': [1e-7]
}

class XGBoost(object):
    def __init__(self, n_estimators=200, learning_rate=0.001, min_samples_split=2,
                 min_impurity=1e-7, max_depth=2):
        self.n_estimators = n_estimators            
        self.learning_rate = learning_rate          
        self.min_samples_split = min_samples_split  
        self.min_impurity = min_impurity              
        self.max_depth = max_depth                  

        self.bar = progressbar.ProgressBar(widgets=utils.bar_widgets)
        
        self.loss = log_loss()

        self.trees = []
        for _ in range(n_estimators):
            tree = XGBoostRegressionTree(
                    min_samples_split=self.min_samples_split,
                    min_impurity=min_impurity,
                    max_depth=self.max_depth,
                    loss=self.loss)

            self.trees.append(tree)

    def fit(self, X, y):
        y = utils.to_categorical(y)

        y_pred = np.zeros(np.shape(y))
        for i in self.bar(range(self.n_estimators)):
            tree = self.trees[i]
            y_and_pred = np.concatenate((y, y_pred), axis=1)
            tree.fit(X, y_and_pred)
            update_pred = tree.predict(X)

            y_pred -= np.multiply(self.learning_rate, update_pred)

    def predict(self, X):
        y_pred = None
        for tree in self.trees:
            update_pred = tree.predict(X)
            if y_pred is None:
                y_pred = np.zeros_like(update_pred)
            y_pred -= np.multiply(self.learning_rate, update_pred)

        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred

