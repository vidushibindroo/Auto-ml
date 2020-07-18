# Auto-ml

Auto-ml is a Automated Machine learning system that automates the data science pipeline process from data pre-processing,
feature selection,algorithm selection, hyper parameter optimization and evaluation.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
Install :
***numpy***
```
pip install numpy

```
Install:
***pandas***
```
pip install pandas

```
# Core Functionality Example

```

from sklearn.data import load_iris
iris = load_iris()
X = iris.data
y = iris.target

from automl import *  # this imports the main automated machine learning interface


ml = AutoML() # create a instance of AutoML class


ml.fit(X, y, test_sz = 0.25)  # Note - type(X) and type(y) should be numpy array
# this returns dataframe score_card

ml.last_score_card  # To see score_card again


ml.get_score_card(name        # saves the score_card as a CSV file
	= "machine_learning_score_card.csv")

```


### DOCUMENTATION

[Decision Tree](https://github.com/vidushibindroo/Auto-ml/blob/master/Documentation/decision_tree_docu.md)
</br>
[Artificial Neural Network](https://github.com/vidushibindroo/Auto-ml/blob/master/Documentation/ANN.md)
</br>
[Naive Bayes](https://github.com/vidushibindroo/Auto-ml/blob/master/Documentation/naivebayes_docu.md)
</br>
[XGBoost](https://github.com/vidushibindroo/Auto-ml/blob/master/Documentation/xgboost_docu.md)
</br>
[Nearest Centroid Classifier](https://github.com/vidushibindroo/Auto-ml/blob/master/Documentation/Nearest_Centroid_Classifier.md)
</br>
[Logistic Regression](https://github.com/vidushibindroo/Auto-ml/blob/master/Documentation/Logisticregression.md)
</br>
[Adaboost Classifier](https://github.com/vidushibindroo/Auto-ml/blob/master/Documentation/Adaboost.md)
</br>
[Linear Discriminant Analysis](https://github.com/vidushibindroo/Auto-ml/blob/master/Documentation/lda.md)
</br>
[Train_test_split](https://github.com/vidushibindroo/Auto-ml/blob/master/Documentation/train_test_docu.md)

















