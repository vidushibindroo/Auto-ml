import numpy as np
import warnings
warnings.filterwarnings("ignore")

class LogisticRegression:

    def __init__(self, alpha=0.0001, n_itr=10000,bias=None):
        self.alpha = alpha
        self.itr = n_itr
        self.weights = None
        self.bias = bias
        

    def sigmoid(self,z):
        return(1/(1+np.exp(-z)))    #Sigmoid function
    

    def fit(self,x,y):   # function performing logistic regression

        self.bias = 0
        self.weights =np.full((x.shape[1],1),0.5)  # Weight initialization

        for i in range(self.itr):  
            z = sigmoid(np.dot(x,self.weights)+ self.bias)         

            dw = np.dot(x.T,(z-np.array([y]).T))  # Gradient Calculation

            self.weights-=(self.alpha*dw)         # Weight updation(Gradient Descent)


    def predict(self,x):

        pred = sigmoid(np.dot(x,self.weights))
        return np.round(pred)