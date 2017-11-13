import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
os.chdir('C:\\Studying\\myvenv\\GrowthHackers\\GH_내부프로젝트(pca,svm)\\DoSVM')

#############################################################################

def SVM_ex(X_train, y_train):
    C_ex= [1e-1, 1, 1e+1]
    gamma_ex= [1e-1, 1, 1e+1]
    clsfic= []
    for C in C_ex:
        for gamma in gamma_ex:
            clsfic.append((C, gamma, SVM_one(X_train, y_train, C= C, gamma= gamma)))
    return clsfic

def best_parameter(X, y):
    C_range = np.logspace(-2, 4, 7)
    gamma_range = np.logspace(-9, 3, 7)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
    grid = GridSearchCV(SVC(kernel= 'rbf'), param_grid=param_grid, cv=cv)
    grid.fit(X, y)
    return grid

def param_express(grid):
    # print('The result is: ', grid.cv_results_)
    print('The best estimator parameter is: ', grid.best_estimator_)
    print('The score of it is: ', grid.best_score_)
    print('The parameters of it is: ', grid.best_params_)

# SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape='ovr', degree=3, gamma=1.0000000000000001e-09,
#   kernel='rbf', max_iter=-1, probability=False, random_state=None,
#   shrinking=True, tol=0.001, verbose=False)
# 0.840136054422
# {'C': 0.01, 'gamma': 1.0000000000000001e-09}

def SVM_one(X_train, y_train, **kwargs):
    if kwargs.get("grid"):
        grid= kwargs.get("grid")
        clf= SVC(C=grid.best_params_['C'], gamma=grid.best_params_['gamma'], kernel='rbf')
        clf.fit(X_train, y_train)
    elif kwargs.get("C") and kwargs.get("gamma"):
        C, gamma= kwargs.get("C"), kwargs.get("gamma")
        clf= SVC(C=C, gamma= gamma, kernel='rbf')
        clf.fit(X_train, y_train)
    else:
        raise IOError
    return clf
