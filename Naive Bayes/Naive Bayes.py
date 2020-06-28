#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from random import randrange
import csv
import math


# In[4]:


def separate(dataset):
    """Separate training set by class value"""
    separated = {}
    for i in range(len(dataset)):
        row = dataset[i]
        if row[-1] not in separated:
            separated[row[-1]] = []
        separated[row[-1]].append(row)
    return separated


# In[5]:


def model(dataset):
    """Find the mean and standard deviation of each feature in dataset"""
    models = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    models.pop() #Remove last entry because it is class value.
    return models


# In[6]:


def model(dataset):
    """finding the mean and standard deviation of each feature in dataset by their class"""
    separated = separate(dataset)
    class_models = {}
    for (classValue, instances) in separated.iteritems():
        class_models[classValue] = model(instances)
    return class_models


# In[7]:


def calculate_pdf(x, mean, stdev):
    """Calculate probability using gaussian density function"""
    if stdev == 0.0:
        if x == mean:
            return 1.0
        else:
            return 0.0
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return 1 / (math.sqrt(2 * math.pi) * stdev) * exponent


# In[8]:


def calculate(models, input):
    """Calculate the class probability for input sample. Combine probability of each feature"""
    probabilities = {}
    for (classValue, classModels) in models.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classModels)):
            (mean, stdev) = classModels[i]
            x = input[i]
            probabilities[classValue] *= calculate_pdf(x, mean, stdev)
    return probabilities


# In[9]:


def predict(models, inputVector):
    """Compare probability for each class. Return the class label which has max probability."""
    probabilities = calculate(models, inputVector)
    (bestLabel, bestProb) = (None, -1)
    for (classValue, probability) in probabilities.iteritems():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


# In[10]:


def getPredictions(models, testSet):
    """Get class label for each value in test set."""
    predictions = []
    for i in range(len(testSet)):
        result = predict(models, testSet[i])
        predictions.append(result)
    return predictions


# In[11]:


def naive_bayes(train, test, ):
    """Create a naive bayes model. Then test the model and returns the testing result."""
    summaries = model(train)
    predictions = getPredictions(summaries, test)
    return predictions

