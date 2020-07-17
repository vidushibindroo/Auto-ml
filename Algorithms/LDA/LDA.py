# Imports

import numpy as np
import pandas as pd
from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import eigh

from utils import data_to_dict



# Linear Discriminant Analysis
class LinearDiscrimentAnalysis:


    def __init__(self, projection_dim):

        if(projection_dim > 1): # add try-except block
            self.projection_dim = projection_dim
        else:
            print("Give Projection Dim > 1")
            
        # initailised these, to keep track of class attributes
        self.W = None
        self.gmeans = None
        self.gcov = None
        self.priors = None
        self.valid_classes = None
        self.num_classes = None
        self.N = None
        
    def fit(self, data, targets, valid_classes = None):
        
        X = data_to_dict(data, targets)
        y = targets

        if valid_classes is None:
            self.valid_classes = np.unique(y)
        else:
            self.valid_classes = valid_classes
        
        self.num_classes = len(self.valid_classes)
        
        means = self.compute_means(X)
        
        # computing scatter matrix
        
        do_scaling = True
        for c, x in X.items():
            if len(x) == 1 or len(x) == 0:
                do_scaling = False
                break
                
        if not do_scaling:
            ismwc = []
            for c, m in means.items():
                sub = np.subtract(X[c], m)
                ismwc.append(np.dot(np.transpose(sub), sub))
            ismwc = np.array(ismwx)
            Sw = np.sum(ismwc, axis = 0) # scatter matrix within class
        
        else:
            class_cov_mat = []
            count = []
            for c, m in means.items():
                sub = np.subtract(X[c], m)
                ni = len(X[c])
                class_cov_mat.append(np.dot(np.transpose(sub), sub)/(ni-1))
                # check if it works properly
            class_cov_mat = np.array(class_cov_mat)
            Sw = np.sum(class_cov_mat, axis = 0)
            
            
        # print("Within class scatter matrix -",Sw)    
        
        class_num = {}
        sum_ = 0
        # print("Lots of debugging print")
        for c, x in X.items():
            # print(x.shape)
            class_num[c] = x.shape[0]
            sum_ += np.sum(x, axis = 0)
        self.N = sum(list(class_num.values()))
        m = sum_/self.N
        
        Sb = []
        for c, mean_class in means.items():
            sub_ = mean_class - m
            Sb.append(np.multiply(class_num[c], np.outer(sub_, sub_.T)))
            
        Sb = np.sum(Sb, axis = 0)
        matrix = np.dot(pinv(Sw), Sb)
        eigen_values, eigen_vectors = eigh(matrix)
        
        eiglist = [(eigen_values[i], eigen_vectors[:, i]) for i in range(len(eigen_values))]
        eiglist = sorted(eiglist, key=lambda x: x[0], reverse=True)
        self.W = np.array([eiglist[i][1] for i in range(self.projection_dim)])
        self.W = np.asarray(self.W).T
        self.g_means, self.g_cov, self.priors = self.gaussian(X)
        return self.W
    
    
    def gaussian(self, X):
        means = {}
        covariance = {}
        priors = {}
        for c, values in X.items():
            proj = np.dot(values, self.W)
            means[c] = np.mean(proj, axis = 0)
            covariance[c] = np.cov(proj, rowvar=False)
            priors[c] = values.shape[0]/self.N
            
        return means, covariance, priors
            
        
    def gaussian_distribution(self, x, u, cov):
        # scaler calculation needs more testing
        scalar = (1. / ((2 * np.pi) ** (x.shape[0] / 2.))) * (1 / np.sqrt(np.linalg.det(cov)))
        x_sub_u = np.subtract(x, u)
        return scalar * np.exp(-np.dot(np.dot(x_sub_u, inv(cov)), x_sub_u.T) / 2.)
    
    def predict(self,X,y):
        proj = self.project(X)
        gaussian_likelihoods = []
        classes = sorted(list(self.g_means.keys()))
        for x in proj:
            row = []
            for c in classes:
                res = self.priors[c] * self.gaussian_distribution(x, self.g_means[c], self.g_cov[c])
                row.append(res)
                
            gaussian_likelihoods.append(row)
        gaussian_likelihoods = np.asarray(gaussian_likelihoods)
        predictions = np.argmax(gaussian_likelihoods, axis=1)
        return predictions
        
        
    def project(self,X):
        # calculates projected values, can be used feature extraction
        return np.dot(X, self.W)
        
        
        
    def compute_means(self, X):
        means = {}
        for c, features in X.items():
            means[c] = np.mean(features, axis = 0)
            
        return means