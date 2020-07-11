from __future__ import division, print_function
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import progressbar
from train_test import train_test_split
from XGBoost.utils import to_categorical, normalize
from XGBoost.utils import mean_squared_error, accuracy_score
from XGBoost.XGB import xgboost_func

def main():
    
    print ("-- XGBoost --")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=2)  

    clf = xgboost_func()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)

if __name__ == "__main__":
    main()