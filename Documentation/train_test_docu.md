# Splitting datasets with train_test_split

The train_test_split function is for splitting a single dataset for two different purposes: training and testing. The training subset is for building your model. The testing subset is for using the model on unknown data to evaluate the performance of the model. With this function, you don't need to divide the dataset manually.

**This is a pythonic implementation of train_test_split using only numpy library.**

### Steps to be followed to run this module:

- Determine dataset to be worked on.
- Load the data as y (the target variable to be predicted) and the rest as X (training data with multiple features).
- Split data using the train_test_split function imported from train_test.py
- Proceed with running the algorithms and rest of the steps following that.



The module train_test contains two functions:
- shuffle_data
- train_test_split

***shuffle_data*** shuffles samples of X and y randomly.

It takes as parameters:
```
X, y:   
    The first parameter is the dataset you're selecting to use.
seed:   
    Seed function is used to save the state of a random function, so that it can generate same 
    random numbers on multiple executions of the code on the same machine or on different machines 
    (for a specific seed value). Default is set as 0, in which case the function will create random partitions, or you can specify a random state for the operation.
```

***train_test_split*** is the main function that splits the dataset into train set and test sets.

It takes as parameters:
```
X, y:   
    Same as mentioned above. 
test_size:
    This parameter specifies the size of the testing dataset. There is no default test size and it 
    needs to be specified while calling the function. It may range from 0.1 to 1.0.
seed:
    Same as mentioned above.
```


## Example code
```
import numpy as np
from sklearn import datasets
from train_test import train_test_split

def main():
    
    print ("--- now running the model ---")

    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=2)  

    #Proceed with calling the algorithms or other functions from here.

if __name__ == "__main__":
    main()
```