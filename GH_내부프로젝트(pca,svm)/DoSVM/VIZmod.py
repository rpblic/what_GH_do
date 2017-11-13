import scipy
import numpy as np
import os
os.chdir('C:\\Studying\\myvenv\\GrowthHackers\\GH_내부프로젝트(pca,svm)\\DoSVM')
import matplotlib.pyplot as plt
import SVMmod, PCAmod

def plot_ex(X_2d, y, clsfic):
    C_ex= [1e-1, 1, 1e+1]
    gamma_ex= [1e-1, 1, 1e+1]
    (X0_min, X0_max, X1_min, X1_max)= PCAmod.minmax(X_2d)
    plt.figure(figsize=(8, 6))
    xx, yy = np.meshgrid(np.linspace(X0_min, X0_max, 200), np.linspace(X1_min, X1_max, 200))
    for (k, (C, gamma, clf)) in enumerate(clsfic):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.subplot(len(C_ex), len(gamma_ex), k + 1)
        plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
                  size='medium')
        # plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
        # plt.contour(xx, yy, Z, colors= ['k', 'k', 'k'], linestyles= ['--', '-', '--'], levels= [-0.5, 0, 0.5])
        contour= plt.contour(xx, yy, Z, colors= 'k', linestyles= '--', levels= [-0.5, 0, 0.5])
        # plt.clabel(contour, fontsize=7, inline= 1)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], s=3, c=y[:,0], cmap=plt.cm.RdBu_r, edgecolors=None)
        # plt.xticks(np.arange(5), [-1.0, -0.5, 0.0, 0.5, 1.0])
        # plt.yticks(np.arange(5), [-1.0, -0.5, 0.0, 0.5, 1.0])
        plt.xticks(np.arange(5))
        plt.yticks(np.arange(5))
        plt.autoscale(enable= True, axis= 'x')
    plt.show()

def plot_heatmap(grid):
    C_range = np.logspace(-2, 4, 7)
    gamma_range = np.logspace(-9, 3, 7)
    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()

def plot_onecase(X_2d, y, **kwargs):
    if kwargs.get('grid'):
        grid= kwargs.get('grid')
        C= grid.best_params_['C']
        gamma= grid.best_params_['gamma']
    elif kwargs.get('clf'):
        clf= kwargs.get('clf')
        C= clf.C
        gamma= clf.gamma
    (X0_min, X0_max, X1_min, X1_max)= PCAmod.minmax(X_2d)
    plt.figure(figsize=(5, 5))
    xx, yy = np.meshgrid(np.linspace(X0_min, X0_max, 200), np.linspace(X1_min, X1_max, 200))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
              size='medium')
    plt.contour(xx, yy, Z, colors= ['k', 'k', 'k'], linestyles= ['--', '-', '--'], levels= [-0.5, 0, 0.5])
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], s=3, c=y[:,0], cmap=plt.cm.RdBu_r,
                edgecolors=None)
    plt.xticks(np.arange(5), [-1.0, -0.5, 0.0, 0.5, 1.0])
    plt.yticks(np.arange(5), [-1.0, -0.5, 0.0, 0.5, 1.0])
    # plt.autoscale(enable= True, axis= 'x')
    plt.xlim(-1.2, 0.5)
    plt.show()
