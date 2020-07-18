import numpy as np
from sklearn import datasets
import sys
import os
from train_test import train_test_split
from Algorithms.DecisionTree.utils import calculate_variance, standardize, calculate_entropy
from Algorithms.XGBoost.utils import mean_squared_error, accuracy_score
from Algorithms.DecisionTree.decision_tree_classification import classification_tree

def main():

    print ("---now running Decision Tree -> Classification---")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=3)

    clf = classification_tree()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

if __name__ == "__main__":
    main()