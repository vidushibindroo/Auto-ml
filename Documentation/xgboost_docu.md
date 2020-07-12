# XGBoost

XGBoost stands for “Extreme Gradient Boosting”, where the term “Gradient Boosting” originates from the paper Greedy Function Approximation: A Gradient Boosting Machine, by Friedman. eXtreme Gradient Boosting or XGBoost is a library of gradient boosting algorithms optimized for modern data science problems and tools. It leverages the techniques mentioned with boosting and comes wrapped in an easy to use library. Some of the major benefits of XGBoost are that its highly scalable/parallelizable, quick to execute, and typically out performs other algorithms.


#### Steps to run the algorithm:

- Determine dataset to be worked on.
- Load the data as y (the target variable to be predicted) and the rest as X (training data with multiple features).
- Split data using the train_test_split function imported from train_test.py
- Run the algorithm on the data.
- Calculate accuracy using the accuracy_score function.


XGBoost has 3 modules:
```
utils
DTreeXGB
XGB
```


### utils

It contains sigmoid_func, bar_widgets, to_categorical, mean_squared_error, accuracy_score and divide.


### DTreeXGB

It contains decision_node and decision_tree.

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
***decision_tree*** is the superclass of the classification tree of XGBoost.

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

### XGB

It contains xgboost_tree, log_loss and xgboost_func.

***xgboost_tree*** is the boosting model of the decision tree ensemble.
It inherits from decision_tree class defined in DTreeXGB. It takes as parameters X and y, that I mentioned above.

***xgboost_func*** is the main function of the XGBoost classification model.

It takes as parameters:
```
n_estimators: int
    The number of classification trees that are used.
    
learning_rate: float
    The step length that will be taken when following the negative gradient during
    training.
    
min_samples_split: int
    The minimum number of samples needed to make a split when building a tree.
    
min_impurity: float
    The minimum impurity required to split the tree further. 
    
max_depth: int
    The maximum depth of a tree.
```

## Example code
```
import numpy as np
from sklearn import datasets
import progressbar
from train_test import train_test_split
from XGBoost.utils import to_categorical, normalize
from XGBoost.utils import mean_squared_error, accuracy_score
from XGBoost.XGB import xgboost_func

def main():
    
    print ("--- now running XGBoost ---")

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
```
