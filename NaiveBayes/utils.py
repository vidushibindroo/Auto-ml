import numpy as np
from random import randrange
import csv
import math


def mean(numbers):
    """Returns the mean of numbers"""
    return np.mean(numbers)


def stdev(numbers):
    """Returns the std_deviation of numbers"""
    return np.std(numbers)


def sigmoid(z):
    """Returns the sigmoid number"""
    return 1.0 / (1.0 + math.exp(-z))


def accuracy_metric(actual, predicted):
    """Calculate accuracy percentage"""
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

