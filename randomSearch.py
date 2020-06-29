import numpy as np
import pandas as pd
# change this with our function
from sklearn.metrics import accuracy_score


class random_search:
    
    """
    Background and Mathematics - 
    
    Class for random search for hyperparameter optimization
    
    Any distribution which has a finite maxima, the maximum of 60 random obseravtions lie within top 5% of the maximum
    with 95% probability.
    
    
    functions - 
    
    1. Constructor (__init__ function)
    
    Input Arguements - 
    - estimator -> the model class, should have fit, predict and scoring class - accuracy
    
    - param_distributions -> dictionary of hyperparamater names as keys and list of values as keys 
    
    - n_iter -> number of iterations random search will run with default value 60
    
    
    2. .fit() method - 
    
    Input Arguements - 
    
    - X_train
    
    - y_train
    
    - X_test
    
    - y_test


    """
    
    def __init__(self, estimator, param_distributions, n_iter = 60):
        
        self.estimator = estimator
        self.num_params = len(param_distributions)
        param_dist = {}
        param_list = []
        for param_tupple_ in param_distributions.items(): # convert to numpy array to optimize work
            param_dist[param_tupple_[0]] = np.array(param_tupple_[1])
            param_list.append(param_tupple_[0])
        self.param_dist = param_dist
        self.param_list = param_list
        
        self.n_iter = n_iter
        
        
        
    def fit(self, X_train, y_train, X_test, y_test):
        
        best_params_ = {}
        best_score_ = 0
        try_params = {}
        score = 0
        
        for i in range(self.n_iter):
            for param in self.param_list:
                try_params[param] = np.random.choice(self.param_dist[param]) # uses normal distribition
            # try_params dictionary is ready now
            try:
                model = self.estimator(**try_params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_pred, y_test)
                if score > best_score_:
                    best_score_ = score
                    best_params_ = try_params.copy()
                
                
            except:
                print('Error in Random Search for Hyperparameter Optimization')
                
        self.best_params_ = best_params_
        self.best_score_ = best_score_



