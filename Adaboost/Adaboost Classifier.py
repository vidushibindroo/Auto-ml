import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score

class DecisionStump():
    
    def __init__(self):
        # Determines if sample shall be classified as -1 or 1 given threshold
        self.polarity = 1
        
        # The index of the feature used to make classification
        self.feature_idx = None
        
        # The threshold value that the feature should be measured against
        self.threshold = None
        
        # Value indicative of the classifier's accuracy
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1

        return predictions


class Adaboost():

    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def fit(self, X, y):
        
        n_samples, n_features = X.shape

        # Initialize weights to 1/N
        w = np.full(n_samples, (1 / n_samples))

        self.clfs = []
        
        for _ in range(self.n_clf):
            clf = DecisionStump()

            min_error = float('inf')
     # Iterate throught every unique feature value and see what value makes the best threshold for predicting y using greedy search 
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    # predict with polarity 1
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    # Error = sum of weights of misclassified samples
                    misclassified = w[y != predictions]
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    # store the best configuration
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error

            # calculate alpha, 1e-10 is added to avoid division by zero
            clf.alpha = 0.5 * np.log((1.0 - min_error) / (min_error + 1e-10))

            # calculate predictions and update weights
            predictions = clf.predict(X)
            
            #Weight updation
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)

            # Save classifiers
            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)

        return y_pred


x=x.values
y=y.values

y[y == 0] = -1
