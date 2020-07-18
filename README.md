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
## Core Functionality Example 1

```

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

from automl import *  # this imports the main automated machine learning interface


ml = AutoML() # create a instance of AutoML class


ml.fit(X, y, test_sz = 0.25)  # Note - type(X) and type(y) should be numpy array
# this returns dataframe score_card

ml.last_score_card  # To see score_card again


ml.get_score_card()     

ml.saveScoreCard(name = "machine_learning_score_card.csv")
 # saves the score_card as a CSV file

```
## Example 2

```
import pandas as pd
data = pd.read_csv('data/Social_Network_Ads.csv') 
X = data.values   ## making X as the features or the input parameters                      
X = X[:,2:4].astype(float)

y = data.values  ## Making Y to contain the output(label) of the data points or features
y = y[:,-1].astype(float)

#Y = df.values  ## Making Y to contain the output(label) of the data points or features
#Y = Y[:,-1]
from automl import AutoML
ml = AutoML()
ml.fit(X, y,test_sz = 0.25)
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

















