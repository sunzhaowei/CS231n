# -*- coding: utf-8 -*-
"""
Created on Sun May 22 10:48:53 2016

@author: iSun
"""

import numpy as np

class KNN(object):
    def __init__(self, K = 3):
        self.K = K
    
    def train(self, X_train, y_train):
        '''
        X: a numpy array of shape (num_train, P)
        y: a numpy array of shape (num_train,)
        '''
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test, num_loops):
        if num_loops == 2:
            dists = self.compute_distances_two_loops(X_test)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X_test)
        elif num_loops == 0:
            dists = self.compute_distances_no_loop(X_test)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)
        
        return self.predict_labels(dists)
    
    def compute_distances_two_loops(self, X_test):
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i,j] = np.sqrt(((X_test[i] - X_train[j])**2).sum())
        return dists
    
    def compute_distances_one_loop(self, X_test):
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))        
        for i in range(num_test):
            dists[i] = np.sqrt(((X_test[i] - self.X_train)**2).sum(1))
        
        return dists
    
    def compute_distances_no_loop(self, X_test):
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        
        EE = (X_test ** 2).sum(axis=1)
        EE2 = np.tile(EE, (num_train, 1)).T
        
        RR = (self.X_train ** 2).sum(axis=1)
        RR2 = np.tile(RR, (num_test, 1))
        
        dists = np.sqrt((EE2 - 2*X_test.dot(X_train.T) + RR2))

        return dists
    
    def predict_labels(self, dists):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        
        orders = dists.argsort(1)
        for i in range(num_test):
            closest_y = [self.y_train[rank] for rank in orders[i][:self.K]]
            y_pred[i] = np.bincount(closest_y).argmax()
        
        return y_pred
        
    
    



#%%
if __name__ == '__main__':
    X_test = np.array([[1,2,3],
                       [2,4,5]])
                       
    X_train = np.array([[3,4,5],
                        [3,0,8],
                        [2,4,5],
                        [1,3,5],
                        [0,5,2]])           
    
    y_train = np.array([1,1,1,0,0])
    
    knn = KNN(K=2)
    knn.train(X_train, y_train)
    print(knn.compute_distances_no_loop(X_test))
    print(knn.predict(X_test, 0))



