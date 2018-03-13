import numpy as np
import os
from sklearn import svm, preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
os.chdir('C:\\Studying\\GrowthHackers\\0819 발표')

"""데이터 열고 전처리, Alcohol과 Malic Acid만 사용"""
with open('wine_data.csv', 'rt') as opener:
    data= opener.readlines()
    data.pop(0)
    data= list(map(lambda x: x[:-1], data))
    data= list(map(lambda x: x.split(','), data))
    data= np.array(data)
    # print(data)
yy= data[:, 0]
XX= data[:, 1:3]
XX= preprocessing.scale(XX)

"""LinearSVC 사용"""
clf= svm.LinearSVC(multi_class='crammer_singer')
clf.fit(XX, yy)
print(clf.coef_, clf.intercept_)
print(clf.decision_function(XX[:5, :]))
print(clf.predict([0,0]))
# print(XX)

"""plt로 class 별로 다른 색으로 점찍기"""
XX1=[]; XX2=[]; XX3=[]
for i, elmt in enumerate(yy):
    if elmt=='1':
        XX1.append(XX[i,:])
    elif elmt=='2':
        XX2.append(XX[i,:])
    elif elmt=='3':
        XX3.append(XX[i,:])
XX1= np.array(XX1); XX2= np.array(XX2); XX3= np.array(XX3)
# print(XX2)
plt.scatter(XX1[:,0],XX1[:,1],c='r', marker='+', label='Class 1')
plt.scatter(XX2[:,0],XX2[:,1],c='g', marker='o', label='Class 2')
plt.scatter(XX3[:,0],XX3[:,1],c='b', marker='*', label='Class 3')

"""함수를 그리기 위한 axis와 mesh 설정"""
ax= plt.gca(); xlim= ax.get_xlim(); ylim= ax.get_ylim()
# print(ax.get_xlim(),ax.get_ylim())
gridx= np.linspace(xlim[0], xlim[1], 100)
gridy= np.linspace(ylim[0], ylim[1], 100)
meshy, meshx= np.meshgrid(gridy, gridx)
gridxy= np.vstack([meshx.ravel(), meshy.ravel()]).T

"""hyperplane 그리기"""
W1= clf.coef_[0]; alpha1, beta1= -W1[0]/W1[1], clf.intercept_[0]/W1[1]
W2= clf.coef_[1]; alpha2, beta2= -W2[0]/W2[1], clf.intercept_[1]/W2[1]
W3= clf.coef_[2]; alpha3, beta3= -W3[0]/W3[1], clf.intercept_[2]/W3[1]
plt.plot(meshx, alpha1*meshx - beta1, 'r--')
plt.plot(meshx, alpha2*meshx - beta2, 'g--')
plt.plot(meshx, alpha3*meshx - beta3, 'b--')

"""plt 설정"""
plt.axis([xlim[0]-0.1, xlim[1]+0.1, ylim[0]-0.1, ylim[1]+0.1])
plt.xlabel('Alcohol')
plt.ylabel('Malic acid')
plt.legend()
plt.title('Wine Class Classification Example_Using SVM')
plt.show()
