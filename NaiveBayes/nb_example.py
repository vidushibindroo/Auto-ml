from __future__ import division, print_function
from sklearn import datasets
import numpy as np
from train_test import train_test_split
from XGBoost.utils import normalize, accuracy_score
from NaiveBayes.NB import naive_bayes

def main():

    print ("--- now running Naive Bayes ---")

    data = datasets.load_digits()
    X = normalize(data.data)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=2)

    clf = naive_bayes()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print ("Accuracy:", accuracy)
    
if __name__ == "__main__":
    main()