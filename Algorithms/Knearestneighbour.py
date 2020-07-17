#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np 
import pandas as pd  
from collections import Counter
import matplotlib.pyplot as plt 


def numpy_distance(x,y):
    return np.linalg.norm(x-y) ### Calculates Euclidean distance between two points



## Empty list which will hold the accuracy values of corresponding K value
## K is a list [1,2,.....34]

def KNN(X_train, X_test, y_train, y_test):
    accuracy = []
    K = np.arange(1,35)
    for k in K:   ## For each value of k we test the model
        prediction = []   ## Prediction of each test case will be stored in this list
        coorect_count = 0  ## how many correct predictions were made
        for i in range(len(X_test)):  ## for each point in testing data we do prediction
            distances = []
            for j in range(len(X_train)):
                dist = numpy_distance(X_test[i], X_train[j])
                ## we are adding the training point, the distance between this training point and
                ## the test point and the label assosciated with the training point
                distances.append((X_train[j], dist, y_train[j]))  
            ## We are sorting this distance list on the basis of distance between the trainin
            ## point and the test point 
            ## key=lambda x: x[1]  here x:x[1] signifies that we will sort this list using 
            ## 2nd(in python indexing starts from 0) column i.e. distance
            distances.sort(key=lambda x: x[1])
            neighbors = distances[:k] ## Considering k closest points
            class_counter = Counter()  ## a counter to check which labels appeared how many times
            for neighbor in neighbors:
                class_counter[neighbor[2]] += 1
            prediction.append(class_counter.most_common(1)[0][0])## whichever appers most we predict that label 
            if(y_test[i] == prediction[i]):  ## if prediction is correct than increase correct count
                coorect_count = coorect_count + 1
        acc = coorect_count/float(len(X_test))  ## accuracy
        accuracy.append(acc)
        return max(accuracy)

    


# In[19]:



# from sklearn.model_selection import train_test_split
# import pandas as pd
# data = pd.read_csv("Social_Network_Ads.csv") 
# X = data.values   ## making X as the features or the input parameters                      
# X = X[:,2:4]

# Y = data.values  ## Making Y to contain the output(label) of the data points or features
# Y = Y[:,-1]

## Splitting the data such that 80% of the data used for training the model and
## 20% data is used to test the modes
# X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2) 
# knns=KNN(X_train, X_test, y_train, y_test) 


# In[6]:





# In[7]:





# In[ ]:





# In[ ]:




