#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

def euclidean_distance(a,b):
    diff = a - b
    return np.sqrt(np.dot(diff, diff))

def load_data(csv_filename):


    return(np.genfromtxt(csv_filename, delimiter=';', skip_header=1)[:,0:11])
    
   
    
def split_data(dataset, ratio = 0.9):
    
    trainingCount=(int(len(dataset)*ratio))
    return((dataset[0:trainingCount],dataset[trainingCount::]))

    
def compute_centroid(data):
   
    return sum(data)/len(data)

    
def experiment(ww_train, rw_train, ww_test, rw_test):

    rw_centroid=compute_centroid(rw_train) 
    ww_centroid=compute_centroid(ww_train)
    
    predictions=0
    correct_predictions=0
    
    for sample in ww_test:
        if(euclidean_distance(ww_centroid,sample)<euclidean_distance(rw_centroid,sample)): ##it's a white wine sample
            correct_predictions+=1
        predictions+=1
    
    for sample in rw_test:
        if(euclidean_distance(rw_centroid,sample)<euclidean_distance(ww_centroid,sample)): ##it's a white wine sample
            correct_predictions+=1
        predictions+=1
        
    print("Total predictions: "+str(predictions))
    print("Correct predictions: "+str(correct_predictions ))
    
    return (correct_predictions/predictions)


def learning_curve(ww_training, rw_training, ww_test, rw_test):
    
    np.random.shuffle(ww_training)
    np.random.shuffle(rw_training)
    
    accuracies=[]
    
    for i in range (0,(min(len(ww_training), len(rw_training)))):
       accuracies.append(experiment(ww_training[:i+1], rw_training[:i+1], ww_test, rw_test))
      
    plt.xlabel("Number of training items used")
    plt.ylabel("accuracy")
    plt.plot(list(range(1,len(accuracies)+1)),accuracies)
     
def cross_validation(ww_data, rw_data, k): 
    
    ww_data_per_partition=int(len(ww_data)/k)
    rw_data_per_partition=int(len(rw_data)/k)
    
    average_sum=0
    ##average_sum=experiment(ww_data[ww_data_per_partition:],rw_data[rw_data_per_partition], ww_data[0:ww_data_per_partition], rw_data[0:rw_data_per_partition])
    
    for i in range(0,k): 
        ww_start=i*ww_data_per_partition
        ww_end=ww_start+ww_data_per_partition
        ww_test=ww_data[ww_start:ww_end]
        ##ww_train=ww_data[:ww_start].reshape(len(ww_data),len(ww_data[0]))+ww_data[ww_end:].reshape(len(ww_data),len(ww_data[0]))
        ##ww_train=ww_data[:ww_start].concatenate(ww_data[ww_end:])   
        ww_train=ww_data[list(range(0,ww_start))+list(range(ww_end,len(ww_data)))]
        
        rw_start=i*rw_data_per_partition
        rw_end=rw_start+rw_data_per_partition
        rw_test=rw_data[rw_start:rw_end]
        ##rw_train=rw_data[:rw_start].concatenate(rw_data[rw_end:])
        ##rw_train=rw_data[:ww_start].reshape(len(rw_data),len(rw_data[0]))+rw_data[ww_end:].reshape(len(rw_data),len(rw_data[0]))
        rw_train=rw_data[list(range(0,rw_start))+list(range(rw_end,len(rw_data)))]
        
        average_sum+=experiment(ww_train, rw_train, ww_test, rw_test)
    
    return(average_sum/k)
    


    
if __name__ == "__main__":
    
    ww_data = load_data('whitewine.csv')
    rw_data = load_data('redwine.csv')

    # Uncomment the following lines for step 2: 
    ww_train, ww_test = split_data(ww_data, 0.9)
    rw_train, rw_test = split_data(rw_data, 0.9)
    experiment(ww_train, rw_train, ww_test, rw_test)
    
    #Uncomment the following lines for step 3
    ww_train, ww_test = split_data(ww_data, 0.9)
    rw_train, rw_test = split_data(rw_data, 0.9)
    learning_curve(ww_train, rw_train, ww_test, rw_test)
    
    # Uncomment the following lines for step 4:
    k = 5
    acc = cross_validation(ww_data, rw_data,k)
    print("{}-fold cross-validation accuracy: {}".format(k,acc))
    


# In[ ]:




