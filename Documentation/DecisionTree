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
