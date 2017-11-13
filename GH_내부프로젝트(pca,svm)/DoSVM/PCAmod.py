import csv
import random
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from random import shuffle, seed
import os
os.chdir('C:\\Studying\\myvenv\\GrowthHackers\\GH_내부프로젝트(pca,svm)\\DoSVM')

def opener():
    with open('..\\voice.csv', 'r') as f:
        #tuple of 3169 data
        #0: male, 1: female
        lines = f.readlines()
        y_at1, X_at1 = [], []
        y_at0, X_at0 = [], []
        for i, line in enumerate(lines):
            if i != 0:
                line = line[:-1]
                splited = line.split(',')
                if splited[0]== '1':
                    y_at1.append([int(splited[0])])
                    X_at1.append([float(value) for value in splited[1:]])
                else:
                    y_at0.append([int(splited[0])])
                    X_at0.append([float(value) for value in splited[1:]])
        length0= len(y_at0)
        length1= len(y_at1)
    random.Random(0).shuffle(X_at1)
    random.Random(0).shuffle(X_at0)
    X_train = np.array(X_at1[int(0.2*length1):] + X_at0[int(0.2*length0):])
    y_train = np.array(y_at1[int(0.2*length1):] + y_at0[int(0.2*length0):])
    X_test = np.array(X_at1[0:int(0.2*length1)] + X_at0[0:int(0.2*length0)])
    y_test = np.array(y_at1[0:int(0.2*length1)] + y_at0[0:int(0.2*length0)])
    return (X_train, y_train, X_test, y_test)

def do_pca(X_train, y_train, X_test, y_test):
    pca = PCA(n_components=10)
    pca.fit(X_train)
    var = pca.explained_variance_ratio_
    var_cum=np.cumsum(np.round(var, decimals=4)*100)
    print(var)
    X_train_trf= scale(pca.transform(X_train))
    X_test_trf = scale(pca.transform(X_test))
    dir(pca)
    comp= pca.components_
    for i in comp:
        list_i= list(i)
        sorted_i= sorted(i, reverse=True)
        max_idx = list_i.index(sorted_i[0])
        print(max_idx)
    return (X_train_trf, y_train, X_test_trf, y_test)

def minmax(X_2d):
    X0_min= X_2d[:,0].min()
    X0_max= X_2d[:,0].max()
    X1_min= X_2d[:,1].min()
    X1_max= X_2d[:,1].max()
    return (X0_min, X0_max, X1_min, X1_max)
##########################################################################################
if __name__== '__main__':
    (X_train, y_train, X_test, y_test)= opener()
    # print(X_train, y_train, X_test, y_test)
    X_2d, y, _, _= do_pca(X_train, y_train, X_test, y_test)
    print(X_2d)
    print(y)
    print(len(X_train), len(y_train), len(X_test), len(y_test))
    print(minmax(X_2d))

# result ############################################################################
# [  9.98674278e-01   1.27360037e-03   4.41226612e-05   5.21008783e-06
#    1.55535256e-06   5.21348397e-07   2.77334192e-07   1.80596617e-07
#    8.87010650e-08   5.88797400e-08]
# 7
# 18
# 9
# 9
# 9
# 19
# 16
# 16
# 5
# 10
# [[-0.07596695 -0.99623148 -0.59891543 ..., -0.8081463  -1.04352789
#    0.38310601]
#  [-0.20018247  4.63839118 -1.03046311 ..., -0.31217125 -0.5552375
#   -0.98400188]
#  [-0.20768527 -0.22487988  0.07060839 ...,  1.21487198 -0.13808354
#    1.02644643]
#  ...,
#  [-0.24074005  0.14397515  1.25745017 ..., -0.37456651  0.03782693
#    0.2239003 ]
#  [-0.15847124 -1.29868753 -0.32369944 ...,  0.49624802  1.7972444
#   -0.04109101]
#  [-0.22362136 -0.40414657  0.71058982 ...,  0.58183959  1.13696501
#   -0.10079734]]
# [[1]
#  [1]
#  [1]
#  ...,
#  [0]
#  [0]
#  [0]]
# 2536 2536 632 632
# (-0.25054937947724226, 9.5204400838559415, -1.5103461944899899, 4.9057691207713772)
