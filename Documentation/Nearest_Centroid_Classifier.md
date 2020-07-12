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







