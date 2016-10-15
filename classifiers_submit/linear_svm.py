# -*- coding: utf-8 -*-
"""
Created on Sun May 22 21:47:29 2016

@author: iSun
"""

import numpy as np

#%%

def L_i(x, y, W):
    delta = 1.0
    scores = W.dot(x)
    D = W.shape[0]
    loss_i = 0.0
    for j in range(D):
        if j == y:
            continue
        loss_i += max(0, scores[j] - scores[y] + delta)
    
    return loss_i

#%%

def L_i_vectorized(x, y, W):
    delta = 1.0
    scores = W.dot(x)
    score_list = np.maximum(0, scores - scores[y] + delta)
    score_list[y] = 0
    loss_i = score_list.sum()
    
    return loss_i

#%%

x = np.array([1,2,3])
y = 1 # [0,1,2,3]

W = np.array([[3,1,4],
              [1,2,0],
              [5,1,2],
              [4,1,3]])
L_i(x, y, W)

L_i_vectorized(x, y, W)



