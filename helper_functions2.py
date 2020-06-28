import numpy as np
import progressbar


bar_widgets = [
    'Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="-", left="[", right="]"),
    ' ', progressbar.ETA()
]

class sigmoid_func():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

    
def ema(running, new, gamma=.9):
##ema is exponential moving average##
    return gamma * running + (1. - gamma) * new


def mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    return mse

def root_mean_squared_error(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return rmse

def mean_absolute_error(y_true, y_pred):
    mse = np.sum(np.absolute(y_true - y_pred))
    return mse

def root_mean_squared_log_error(y_true, y_pred):
    rmsle = np.sqrt(np.square(np.log(y_true + 1) - np.log(y_pred + 1)).mean())
    return rmsle

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy
