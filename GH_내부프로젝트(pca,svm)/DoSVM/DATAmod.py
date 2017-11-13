import scipy
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
os.chdir('C:\\Studying\\myvenv\\GrowthHackers\\GH_내부프로젝트(pca,svm)\\DoSVM')

def opener():
    with open('..\\PREPROCESSED_WA_Fn-UseC_-HR-Employee-Attrition.csv', 'rt') as opener:
        data= opener.readlines()
        data.pop(0)
        data= list(map(lambda x: x[:-1], data))
        data= list(map(lambda x: x.split(','), data))
        data= np.array(data)
    y = data[:, 1]
    X_scaled = np.insert(data[:,2:], 0, data[:,0], axis=1)
    # X_2d= X_scaled[:,(17,31)]
    # return data
    # scaler= StandardScaler()
    # X_scaled = scaler.fit_transform(X_scaled)
    return (X_scaled, y)

##########################################################################
if __name__== "__main__":
    print(opener()[0][:,(17,31)])
