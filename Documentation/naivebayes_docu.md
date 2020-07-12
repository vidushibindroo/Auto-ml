# Naive Bayes

A Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task. The crux of the classifier is based on the Bayes theorem.


Steps to run the algorithm:

- Load the data as y (the target variable to be predicted) and the rest as X (training data with multiple features).
- Split data using the train_test_split function imported from train_test.py
- Run the algorithm on the data.
- Calculate accuracy using the accuracy_score function.


NaiveBayes has 2 modules:
```
utils
NB
```


#### utils

It contains mean, stdev, sigmoid and accuracy_metric, that will return the mean of numbers, standard deviation of numbers, sigmoid number and accuracy percentage, respectively.


#### NB

It contains likelihood, prior and classification functions other than fit and predict functions to form the Gaussian Naive Bayes classifier.

***likelihood*** calculates the Gaussian likelihood of the data x given mean and var and takes the same as its parameters.

***prior*** calculates the prior of class c (samples where class == c / total number of samples), taking the same as its parameter.

***classification*** is the main function. It classifies using Bayes Rule P(Y|X) = P(X|Y)*P(Y)/P(X) or Posterior = Likelihood * Prior / Scaling Factor

P(Y|X) - The posterior is the probability that sample x is of class y given the
        feature values of x being distributed according to distribution of y and the prior.
        
P(X|Y) - Likelihood of data X given class distribution Y.
        Gaussian distribution (given by likelihood)
        
P(Y)   - Prior (given by prior)

P(X)   - Scales the posterior to make it a proper probability distribution.
        This term is ignored in this implementation since it doesn't affect
        which class distribution the sample is most likely to belong to.
        Classifies the sample as the class that results in the largest P(Y|X) (posterior)

We used a naive assumption (independence): P(X1,X2,X3|y) = P(X1|y)*P(X2|y)*P(X3|y)
Posterior is product of prior and likelihoods (ignoring scaling factor).
The classify function returns a class with the largest posterior probability.

***predict*** function predicts the class labels of the samples in X.


## Example code
```
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
```
