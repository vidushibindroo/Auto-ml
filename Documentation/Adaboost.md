# Adaboost Classifier

AdaBoost, short for “Adaptive Boosting” focuses on classification problems and aims to convert a set of weak classifiers into a strong one.

#### Running the algorithm:

-Determine dataset to be worked on

-Split data using the train_test_split function imported from train_test.py 

-Load the data as y (the target variable to be predicted) and the training data as x 

-Run the algorithm on the data

-Calculate accuracy/f1_score using the evaluation metrics





#### Methods

***fit*** method takes the training data as arguments

***predict*** method predicts the class labels of the samples in x

## Example code
```
import numpy as np

clf = Adaboost(#hyperparameter)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Adaboost classification accuracy and f1-score:", accuracy_score(y_test, y_pred),f1_score(y_test,y_pred))
```
