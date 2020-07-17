# Logistic Regression

Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable.

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

clf = LogisticRegression(#hyper-parameters)
clf.fit(x, y)
predictions = clf.predict(x)

print("LR classification accuracy:", accuracy_score(y, predictions),f1_score(y,predictions))
```
