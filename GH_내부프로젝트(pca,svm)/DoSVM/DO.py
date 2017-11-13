import DATAmod, SVMmod, VIZmod, PCAmod
from sklearn.svm import SVC

(X_train, y_train, X_test, y_test)= PCAmod.opener()
X_train, y_train, X_test, y_test= PCAmod.do_pca(X_train, y_train, X_test, y_test)
clsfic= SVMmod.SVM_ex(X_train, y_train)

for i in clsfic:
    print(i[2].score(X_test, y_test))

# VIZmod.plot_ex(X_test, y_test, clsfic)

grid= SVMmod.best_parameter(X_train, y_train)
SVMmod.param_express(grid)
VIZmod.plot_heatmap(grid)
# best_clf= SVMmod.SVM_one(X_train, y_train, grid=grid)
# best_clf= SVMmod.SVM_one(X_train, y_train, C= 10, gamma= 10)
# VIZmod.plot_onecase(X_test, y_test, grid, best_clf)
# VIZmod.plot_onecase(X_test, y_test, clf= best_clf)
