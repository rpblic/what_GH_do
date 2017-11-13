import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
import os
os.chdir('C:\\Studying\\GrowthHackers\\0808ML')

'''데이터 꺼내서 ndarray로 정렬'''
opener= open('newtrain.csv', 'rt')
data= opener.readlines()
for i, elmt in enumerate(data):
    elmt= elmt[:-1]
    elmt= elmt.split(',')
    # print(elmt)
    data[i]= elmt

data= np.array(data)
# print(data)
Xdat= data[:,0]; Ydat= data[:,1]
print(Xdat.sum())
# def Jfunc(tta0,tta1):

'''점찍고 선 긋기'''
# plt.plot(Xdat, Ydat, 'k*')
# plt.show()
