# Decision Tree

Decision tree builds classification or regression models in the form of a tree structure. It breaks down a 
dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally 
developed. The final result is a tree with decision nodes and leaf nodes.


#### Steps to run the algorithm:

- Determine dataset to be worked on.
- Load the data as y (the target variable to be predicted) and the rest as X (training data with multiple features).
- Split data using the train_test_split function imported from train_test.py
- Run the algorithm on the data.
- Calculate accuracy using the accuracy_score function.


DecisionTree has 3 modules:
```
utils
DTreeXGB
decision_tree_classification
```


### utils

It contains calculate_entropy, calculate_variance, standardize (from Algorithms.DecisionTree.utils) and mean_squared_error, accuracy_score and divide (from Algorithms.XGBoost.utils).


### DTreeXGB

It contains decision_node and decision_tree and is the same one used for XGBoost.

***decision_node*** is just the class that represents a decision node or leaf in the decision tree.

It takes as parameters:
```
feature_i: int
    Feature index which we want to use as the threshold measure.
    
threshold: float
        The value that we will compare feature values at feature_i against to
        determine the prediction.
        
value: float
        The class prediction if classification tree.
        
true_branch: decision_node
        Next decision node for samples where features value met the threshold.
        
false_branch: decision_node
        Next decision node for samples where features value did not meet the threshold.
```
***decision_tree*** is the superclass of the classification tree of Decision Tree.

It takes as parameters:
```
min_samples_split: int
    The minimum number of samples needed to make a split when building a tree.
    
min_impurity: float
    The minimum impurity required to split the tree further.
    
max_depth: int
    The maximum depth of a tree.
    
loss: function
    Loss function that is used for Gradient Boosting models to calculate impurity.
```

### decision_tree_classification

It contains classification_tree and inherits from decision_tree class defined in DTreeXGB.

***information_gain*** calculates the information gain.
Information gain is calculated based on the decrease in entropy after a dataset is split on 
an attribute. This is essential for feature/attribute selection as the feature that yields maximum
reduction in entropy would provide maximum information about Y.
It takes as parameters y, y1 and y2.

***majority_vote*** is the method by which label is assigned to each leaf in a classification tree.
It takes as parameter y.

## Example code
```
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
```