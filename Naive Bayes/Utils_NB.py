#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from random import randrange
import csv
import math


# In[2]:


def mean(numbers):
    """Returns the mean of numbers"""
    return np.mean(numbers)


# In[3]:


def stdev(numbers):
    """Returns the std_deviation of numbers"""
    return np.std(numbers)


# In[4]:


def sigmoid(z):
    """Returns the sigmoid number"""
    return 1.0 / (1.0 + math.exp(-z))


# In[5]:


def accuracy_metric(actual, predicted):
    """Calculate accuracy percentage"""
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

