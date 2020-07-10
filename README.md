# Auto-ml

Auto-ml is a Automated Machine learning system that automates the data science pipeline process from data pre-processing,
feature selection,algorithm selection, hyper parameter optimization and evaluation.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
Install :
***numpy***
```
pip install numpy

```
Install:
***pandas***
```
pip install pandas

```
# Core Functionality Example
??????? //write the code



<br />
<br />
<br />
<br />

## Decision Tree

Steps to apply algorithm:

- Read csv file using pandas
- Split the data into training and test sets
- Use decision tree algorithm 
- Find the accuracy

### Load and Prepare Data
Format of the data:

    - the last column of the data frame must contain the label and it must also be called "label".
    - there should be no missing values in the data frame.

Decision Tree has two modules:


    - helper_functions
    - decision_tree_functions
 
**helper_functions**

It consists of generate_data, create_plot, train_test_split

*how to use it?*

### from helper_functions import generate_data, create_plot, train_test_split

1. train_test_split(df, test_size)
It takes two inputs:
    1. the csv file 
    2. test size 
    
It will divide data into test set and train set i.e returns train and test dataframe

### def train_test_split(df, test_size):
    
   if isinstance(test_size, float):     
   ## it will check whether test size is in proportion. if its in proportion it will count the rows tests datafrme will contain
        test_size = round(test_size * len(df))
    indices = df.index.tolist()     
   ## As random.sample function takes list of indices and k is the no.of rows that test size will contain 
    test_indices = random.sample(population=indices, k=test_size)

   test_df = df.loc[test_indices]   
   ## loc function will add rows having test indices
   train_df = df.drop(test_indices)  
   ## add other rows to train dataframe
    
   return train_df, test_df



**decision_tree_functions**
```
 example_tree = {"petal_width <= 0.8": ["Iris-setosa", 
                                      {"petal_width <= 1.65": [{"petal_length <= 4.9": ["Iris-versicolor", 
                                                                                        "Iris-virginica"]}, 
                                                                "Iris-virginica"]}]}
```
```
sub_tree = {"question": ["yes_answer", 
                         "no_answer"]}
```


### decision_tree_algorithm(train_df, ml_task="classification", max_depth=5)

It takes 3 inputs train dataframe,ml_task: either regression or classification and max_depth which is the integer varible (the depth of tree)

## Example code
```
import numpy as np
import pandas as pd
from randomSearch import random_search 
from pprint import pprint

from decision_tree_functions import decision_tree_algorithm, make_predictions, calculate_accuracy
from helper_functions import generate_data, create_plot, train_test_split

df = pd.read_csv("Iris.csv")
df = df.drop("Id", axis=1)
df = df.rename(columns={"species": "label"})
X=pd.read_csv("Iris.csv")
y=X["species"]
X.drop(["Id","species"],axis=1, inplace=True)
train_df, test_df = train_test_split(df, test_size=20)
tree = decision_tree_algorithm(train_df, ml_task="classification", max_depth=3)
pprint(tree)
accuracy = calculate_accuracy(test_df, tree)
print(accuracy)
```

<br />
<br />
<br />
<br />





## ARTIFICIAL NEURAL NETWORK

- Define independent variables and dependent variable
- Define Hyperparameters
- Define Activation Function and its derivative
- Train the model
- Make predictions

Class MLP implements a multi-layer perceptron (MLP) algorithm that trains using Backpropagation.

**MLP class has following functions:**

## init(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2)
  *Constructor for the MLP. Takes the number of inputs,
        a variable number of hidden layers, and number of outputs
        Args:
        num_inputs (int): Number of inputs
        hidden_layers (list): A list of ints for the hidden layers
        num_outputs (int): Number of outputs*
## forward_propagate(self, inputs):
  *Computes forward propagation of the network based on input signals.
       Args:
       inputs (ndarray): Input signals
       Returns:
       activations (ndarray): Output values*
## back_propagate(self, error):
 *Backpropogates an error signal.
    Args:
    error (ndarray): The error to backprop.
    Returns:
    error (ndarray): The final error of the input*
## train(self, inputs, targets, epochs, learning_rate):
  *Trains model running forward prop and backprop
    Args:
    inputs (ndarray): X
    targets (ndarray): Y
    epochs (int): Num. epochs we want to train the network for
    learning_rate (float): Step to apply to gradient descent*
## gradient_descent(self, learningRate=1):
  *Learns by descending the gradient
    Args:
    learningRate (float): How fast to learn.*

* Activation Function and it’s derivative: Our activation function is the sigmoid function.*
## _sigmoid(self, x):
  *Sigmoid activation function
    Args:
    x (float): Value to be processed
    Returns:
    y (float): Output*
## sigmoidderivative(self, x):
  *Sigmoid derivative function
    Args:
    x (float): Value to be processed
    Returns:
    y (float): Output*
## _mse(self, target, output):
  *Mean Squared Error loss function
    Args:
    target (ndarray): The ground trut
    output (ndarray): The predicted values
    Returns:
        (float): Output*
### Example Code
```
    from ann import MLP
    # create a dataset to train a network for the sum operation
    items = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])

    # create a Multilayer Perceptron with one hidden layer
    mlp = MLP(2, [5], 1)

    # train network
    mlp.train(items, targets, 50, 0.1)

    # create dummy data
    input = np.array([0.3, 0.1])
    target = np.array([0.4])

    # get a prediction
    output = mlp.forward_propagate(input)

    print()
    print("Our network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))
```

<br />
<br />
<br />
<br />





## Nearest-Centroid-Classifier
In machine learning, a nearest centroid classifier or nearest prototype classifier is a classification model that assigns to observations the label of the class of training samples whose mean (centroid) is closest to the observation.
<img>![image.png](attachment:image.png)</img>
### euclidean_distance
The Euclidean distance is the "ordinary" straight-line distance between two points in Euclidean space
### load_data(csv_filename)
```
Returns a numpy ndarray in which each row repersents
a wine and each column represents a measurement. There should be 11
columns (the "quality" column cannot be used for classificaiton).
"""
"""
file=open(csv_filename, 'r')
temp=[]

rowCount=0
file.readline()

for line in file:
    row=line.split(";")[0:11]
    temp.append(row)
    rowCount+=1

return np.array(temp)
```
### split_data(dataset, ratio = 0.9)

*Return a (train, test) tuple of numpy ndarrays. 
The ratio parameter determines how much of the data should be used for 
training. For example, 0.9 means that the training portion should contain
90% of the data. You do not have to randomize the rows. Make sure that 
there is no overlap. 

### compute_centroid(data):

*Returns a 1D array (a vector), representing the centroid of the data
    set. 
 
### experiment(ww_train, rw_train, ww_test, rw_test): 

*Train a model on the training data by creating a centroid for each class.
    Then test the model on the test data. Prints the number of total 
    predictions and correct predictions. Returns the accuracy.*
    
    IS THIS RIGHT?? IS IT TRAINING DATA?
    
Write the function learning_curve(ww_training, rw_training, ww_test, rw_test) which performs the following steps:

Shuffle the two training sets. Run n training/testing experiments (by using the experiment function from part 2), where n, is the size of the smaller one of the training sets (with this specific data set, the two classes contain the same amount of data items).

For each experiment, increase the size of each training data set by one. In the first call to experiment, you would only use the first data item from each training data set. In the second call you use the first two data items in each training set etc. Always use the full testing set. Collect the accuracies returned by each experiment in a list. Use matplotlib to plot a graph in which the x-axis is the number of training items used and the y-axis is the accuracy. 

### learning_curve(ww_training, rw_training, ww_test, rw_test):

*Perform a series of experiments to compute and plot a learning curve.*

### cross_validation(ww_data, rw_data, k)

*Perform k-fold crossvalidation on the data and print the accuracy for each
fold. 
##Will k will never be greater than number of data entries. Remainder should be added to training data
nested [[d],[d]]*

## Example code

``` 
    from Nearestcentroidclassification import load_data,split_data,experiment,learning_curve,cross_validation 
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
```    










