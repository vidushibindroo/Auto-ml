import time
import numpy as np
import pandas as pd
from train_test import train_test_split

# import other classes
from Algorithms.NaiveBayes.utils import *
from Algorithms.NaiveBayes.NB import *
from Algorithms.XGBoost.utils import *
from Algorithms.XGBoost.XGB import *
from Algorithms.XGBoost.DTreeXGB import *
from Algorithms.DecisionTree.utils import *
from Algorithms.DecisionTree.decision_tree_classification import classification_tree
from HelperFunctions.helper_functions import dt_train_test_split
from Algorithms.Knearestneighbour import KNN,numpy_distance
from Algorithms.ArtificialNeuralNetwork import MLP
from Algorithms.LDA.LDA import *
from randomSearch import *
#from Algorithms.DecisionTree import dtreeRandomSearch
from Algorithms.Nearestcentroidclassification import *
from Algorithms.LogisticRegression import *

from Algorithms.Adaboost.AdaboostClassifier import *



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
adb_dic={'n_clf':[1,2,3,5,7,10,12,15,17,20, 100]}


class AutoML():
    
    
    """
    
    This class acts as the main interface for AutoML module.
    
    """
    
    
    
    def __init__(self, problem_type = 'classification'):
        self.problem_type = problem_type
        self.last_score_card = None
        print("AutoML Instance Initiated")
        
        
        
    def fit(self, features, target_variable, test_sz = 0.25, seed = 2, threshold = None):
        
        """
        Runs all classification algorithms turn by turn and Pandas dataframe as score card with classification accuracy
        
        """
        
        
        print("---AutoML Running ---\n\n")
        X = features
        y = target_variable
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, seed=seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_sz)
        
        num_class = len(np.unique(target_variable))
        
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
        if X.shape[1] != 1:
            clf = LinearDiscrimentAnalysis(projection_dim=2)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test,y_test)
            score = accuracy_score(y_test, pred)
            params = lda_dic
            scoreCard.append({
                            'Algorithm': 'Linear Discriminant Analysis',
                            'Accuracy Score': score,
                             'Params':{'projection_dim': 2}
                             })
            best_so_far_ = max(best_so_far_, score)
            del clf


        #############################################################################
        # Naive Bayes

        clf = naive_bayes()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scoreCard.append({
                        'Algorithm': 'Naive Bayes',
                        'Accuracy Score': score,
                         'Params':'-'
                         })
        best_so_far_ = max(best_so_far_, score)

        #############################################################################
        
        # logistic Regression
        
        
        rs=random_search(LogisticRegression,lr_dic, n_iter = 300)
        rs.fit(X_train,y_train,X_test,y_test)
        scoreCard.append({
                        'Algorithm': 'Logistice Regression',
                        'Accuracy Score': rs.best_score_,
                         'Params':rs.best_params_
                         })
        best_so_far_ = max(best_so_far_, rs.best_score_)
        del rs


        #############################################################################
        
        # KNN
        
        clf=KNN(X_train, X_test, y_train, y_test)
        score=clf
        best_so_far_=max(best_so_far_,score)   
        scoreCard.append({
                        'Algorithm': 'KNN',
                        'Accuracy Score': score,
                         'Params': '-'
                         })
        del clf
        
        #############################################################################

        #Decision Tree

        clf = classification_tree()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = accuracy_score(y_test, y_pred)

        scoreCard.append({
                        'Algorithm': 'Decision Tree',
                        'Accuracy Score': score,
                         'Params':{'min_samples_split': 2}
                         })
        best_so_far_ = max(best_so_far_, score)

        #############################################################################
        
        # dtree
        
        #df = pd.concat([pd.DataFrame(X), pd.DataFrame(y, columns = ['label'])], axis = 1)
        #train_df, test_df = dt_train_test_split(df, test_size=20)
        #rs = random_search_dtree(decision_tree_algorithm, dtree_dic, n_iter = 8)
        #rs.fit(train_df, test_df)
        #scoreCard.append(['Decision Tree', rs.best_score_, rs.best_params_])
        #scoreCard.append({
        #               'Algorithm': 'Decision Tree',
        #                'Accuracy Score': rs.best_score_,
        #                 'Params': rs.best_params_
        #                 })
        #best_so_far_ = max(best_so_far_, rs.best_score_)
        #del df # delete dataframe to clear ram
        
        
        #############################################################################
        
        # Random Forrest
        

        
        
    
        
        #scoreCard.append(['Random Forrest', rs.best_score_, rs.best_params_])
        #best_so_far_ = max(best_so_far_, rs.best_score_)
        
        #############################################################################
        
        
        
        if(threshold is not None and best_so_far_ >= threshold):
            score_card = self.get_score_card(scoreCard)
            return score_card
            
        
        
        
        # Further Algorithms Only used when best_so_far_ is still less than threshhold,
        # by default, it is set to 'None'
        
        #############################################################################
        
        # XGB
        
        rs = random_search(xgboost_func, xgb_dic, n_iter = 1 ) # 50)
        rs.fit(X_train, y_train, X_test, y_test) 
        #scoreCard.append(['XGBoost', rs.best_score_, rs.best_params_])
        scoreCard.append({
                        'Algorithm': 'XGBoost',
                        'Accuracy Score': rs.best_score_,
                         'Params':rs.best_params_
                         })
        best_so_far_ = max(best_so_far_, rs.best_score_)
        del rs
        #############################################################################
        
        # Adaboost
        
        rs=random_search(Adaboost,adb_dic, n_iter = 7)
        rs.fit(X_train,y_train,X_test, y_test)
        #scoreCard.append(['Adaboost', rs.best_score_, rs.best_params_])
        scoreCard.append({
                        'Algorithm': 'Adaboost',
                        'Accuracy Score': rs.best_score_,
                         'Params':rs.best_params_
                         })
        best_so_far_ = max(best_so_far_, rs.best_score_)
        del rs

        #############################################################################
        
        
        score_card = self.get_score_card(scoreCard)
        return score_card
        
    
    def get_score_card(self, scoreCard = None, SortBy = 'Accuracy Score'):
        
        """
        Return Score card as a Pandas dataframe
        
        """
        
        Col = ['Algorithm', 'Accuracy Score', 'Params']
        score_card = pd.DataFrame(columns = Col)
        if scoreCard is None:
            return score_card
        else:
            for i in scoreCard:
                score_card = score_card.append(i, ignore_index = True)
            score_card = score_card.sort_values(by=['Accuracy Score'], ascending=False)
            score_card = score_card.reset_index(drop = True)    
        self.last_score_card = score_card
        return score_card    
    
    
    
    def saveScoreCard(self, name = "score_card.csv"):
        self.last_score_card.to_csv(name)
        print('File Saved')
        return
