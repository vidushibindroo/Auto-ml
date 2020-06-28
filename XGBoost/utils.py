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
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse


def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

def to_categorical(y):
    m = np.shape(y)[0]
    col = np.max(y) + 1
    one_hot = np.zeros((m, col))
    one_hot[np.arange(m), y] = 1
    return one_hot