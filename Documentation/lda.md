# LDA: Linear Discrimant Analysis

Linear discriminant analysis (LDA), normal discriminant analysis (NDA), or discriminant function analysis is a generalization of Fisher's linear discriminant, a method used in statistics, pattern recognition, and machine learning to find a linear combination of features that characterizes or separates two or more classes of objects or events.

### utils

It contains data_to_dict.

### LinearDiscrimentAnalysis

***Constructor:*** parameter - projection_dim

***fit:*** 

data

targets

valid_classes = None

***gaussian:*** 
X


***gaussian_distribution:***
x

u

cov

***predict:***
X

y

***project:***
X

***compute_means:***
X


***Example:***


from sklearn.metrics import accuracy_score

```
clf = LinearDiscrimentAnalysis(projection_dim=4)
clf.fit(X_train.values, y_train)
pred = clf.predict(X_test,y_test)
accuracy_score(y_test, pred)
```
