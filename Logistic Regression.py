#Logistic Regression

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def sigmoid(z):
    return(1/(1+np.exp(-z)))    #Sigmoid function

def log_reg(x_train,y_train,alpha,itr):   # function performing logistic regression
    
    m=x_train.shape[0]  
    c=0
    w=0
    w=np.full((x_train.shape[1],1),0.5)  # Weight initialization
    
    for i in range(itr):  
        z = sigmoid(np.dot(x_train,w))         

        dw = np.dot(x_train.T,(z-np.array([y]).T))  # Gradient Calculation

        w-=(alpha*dw)         # Weight updation(Gradient Descent)
        
    return w

def predict(x_test):
    
    pred = sigmoid(np.dot(x_test,w))
    
    return pred