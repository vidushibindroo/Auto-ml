import time
import numpy as np
import pandas as pd
from train_test import train_test_split

# import other classes
from Algorthms.NaiveBayes.utils import *
from Algorithms.NaiveBayes.NB import *
from Algorithms.XGBoost.utils import *
from Algorithms.XGBoost.XGB import *
from Algorithms.XGBoost.DTreeXGB import *
from Algorithms import *
import progressbar
from Algorithms.DecisionTree.decision_tree_functions import decision_tree_algorithm, make_predictions, calculate_accuracy
from HelperFunctions.helper_functions import generate_data, create_plot, train_test_split
from Algorithms.Knearestneighbour import KNN,numpy_distance
from pprint import pprint
from collections import Counter
import matplotlib.pyplot as plt 
from random import random
from Algorithms.ArtificialNeuralNetwork import MLP
from LDA.LDA import *
import randomSearch
from Algorithms.DecisionTree import dtreeRandomSearch
from Algorithms.Nearestcentroidclassification import *






# hyperparameter distributions for each algorithm


lda_dic = {
    'projection_dim': 2
}

# logistic Regression
lr_dic={'alpha': [0.001,0.005,0.01,0.03,0.05,0.07,0.1],
        'n_itr':[750,1000,1250,1500,1750,2000],
        'bias':[0,5,10,15,25,50,75,100]}


# dtree

dtree_dic = {
    'ml_task':["classification"],
    'counter': [0],
    'min_samples': [1, 2, 3],
    'max_depth': [3, 4, 5, 6, 7]
}


#Xgb
xgb_dic = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'min_samples_split': [1, 2, 3],
    'max_depth': [2, 3, 4, 5, 6, 7],
    'min_impurity': [1e-7]
}


#adaboost
adb_dict={'n_clf':[3,5,7,10,12,15,17,20]}


class AutoML():
    
    
    """
    
    This class acts as the main interface for AutoML module.
    
    """
    
    
    
    def __init__(self, problem_type = 'classification'):
        self.problem_type = problem_type
        
        self.last_score_card = None
        
        
        
        
        
        
    def fit(self, features, target_variable, test_size = 0.25, seed = 2, threshold = None):
        
        """
        Runs all classification algorithms turn by turn and Pandas dataframe as score card with classification accuracy
        
        """
        
        
        
        X = features
        y = target_variable
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, seed=seed)
        
        
        num_class = np.unique(target_variable)
        
        if num_class == 1:
            print("Invalid Data Alert : Only one target class")
            return
        
        if num_class == 2:
            print("-- Begin Binary Classification --")
            
        if num_class > 2: 
            print("-- Begin Multi Class Classication --")
            print("Number of detected classes : ", num_class )
            
        
        
        
        best_so_far_ = 0
        scoreCard = [] # initialised empty dictionary
        
        
        
        
        # Linear Algorithms - Fastest
        
        #############################################################################
        # lda 
        
        clf = LinearDiscrimentAnalysis(projection_dim=2)
        clf.fit(X_train.values, y_train)
        pred = clf.predict(X_test,y_test)
        score = accuracy_score(y_test, pred)
        params = lda_dic
        scoreCard.append(['Linear Discriminant Analysis', score, lda_params])
        best_so_far_ = max(best_so_far_, score)
        
        #############################################################################
        # Naive Bayes

        clf = naive_bayes()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scoreCard.append(['Naive Bayes', score, '-'])
        best_so_far_ = max(best_so_far_, score)

        #############################################################################
        
        # logistic Regression
        
        
        rs=random_search(LogisticRegression,lr_dic)
        rs.fit(x_train,y_train,x_test,y_test)
        scoreCard.append(['Logistic Regression', rs.best_score_, rs.best_params_])
        best_so_far_ = max(best_so_far_, rs.best_score_)
        
        #############################################################################
        
        # KNN
        
        accuracy = []
        K = np.arange(1,35)
        clf=KNN(X_train, X_test, y_train, y_test)
        score=max(accuracy)
        best_so_far_=max(best_so_far_,score)
        scoreCard.append(['KNN', score, '-'])
        
        
        
        
        
        #############################################################################
        
        # dtree
        
        rs = random_search_dtree(decision_tree_algorithm, dtree_dic, n_iter = 8)
        rs.fit(train_df, test_df)
        scoreCard.append(['Decision Tree', rs.best_score_, rs.best_params_])
        best_so_far_ = max(best_so_far_, rs.best_score_)
        
        
        
        #############################################################################
        
        # Random Forrest
        
        
        
        
        
        scoreCard.append(['Random Forrest', rs.best_score_, rs.best_params_])
        best_so_far_ = max(best_so_far_, rs.best_score_)
        
        #############################################################################
        
        
        
        if(threshold is not None and best_so_far_ >= threshold):
            score_card = get_score_card(scoreCard)
            return score_card
            
        
        
        
        # Further Algorithms Only used when best_so_far_ is still less than threshhold,
        # by default, it is set to 'None'
        
        #############################################################################
        
        # XGB
        
        rs = random_search(xgboost_func, xgb_dic, n_iter = 50)
        rs.fit(X_train, y_train, X_test, y_test) 
        scoreCard.append(['XGBoost', rs.best_score_, rs.best_params_])
        best_so_far_ = max(best_so_far_, rs.best_score_)
        
        #############################################################################
        
        # Adaboost
        
        rs=random_search(Adaboost,adb_dict, n_iter = 5)
        rs.fit(X_train,y_train,X_test, y_test)
        scoreCard.append(['Adaboost', rs.best_score_, rs.best_params_])
        best_so_far_ = max(best_so_far_, rs.best_score_)
        
        #############################################################################
        # Neural Networks add unwanted complexity for simpler problems hence added in last
        # ANN
        
        
        #scoreCard.append(['ANN', , ])
        
        #############################################################################
        
        
        score_card = get_score_card(scoreCard)
        return score_card
        
        
        
        
        
        
    def get_score_card(self, scoreCard = None, SortBy = 'Accuracy Score'):
        
        """
        Return Score card as a Pandas dataframe
        
        """
        
        Col = ['Algorithm', 'Accuracy Score', 'Params']
        
        if scoreCard is None:
            score_card = pd.DataFrame(columns = Col)
        else:
            score_card = pd.concat([pd.DataFrame(i, columns = Col) for i in scoreCard],
              ignore_index=True)
            score_card.sort_values(by=['Accuracy Score'])
            score_card.reset_index(drop = True)    
        self.last_score_card = score_card
        return score_card
    
    
    
    def saveScoreCard(self, name = "score_card.csv"):
        self.last_score_card.to_csv(name)
        print('File Saved')
        return
