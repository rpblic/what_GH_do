import DATAmod, SVMmod, VIZmod, PCAmod
from sklearn.svm import SVC

(X_train, y_train, X_test, y_test, attributes)= PCAmod.opener('..\\PREPROCESSED_WA_Fn-UseC_-HR-Employee-Attrition.csv')
X_train, y_train, X_test, y_test= PCAmod.do_pca(X_train, y_train, X_test, y_test)
# X_train, y_train, X_test, y_test= PCAmod.just_scale(X_train, y_train, X_test, y_test)
# X_train, y_train, X_test, y_test, X_att_list= PCAmod.get_param(X_train, y_train, X_test, y_test, attributes)
# for k, (trainset, testset) in enumerate(zip(X_train, X_test)):
#     clsfic= SVMmod.SVM_ex(trainset, y_train)
#     print('for Case when {}:\n'.format(str(X_att_list[k])))
#     for i in clsfic:
#         print('\tWhen C is {} and gamma is {}, the score of prediction is: {:.6f}\
#         '.format(i[0], i[1], i[2].score(testset, y_test)))
#     VIZmod.plot_ex(testset, y_test, clsfic, label= X_att_list[k])

clsfic= SVMmod.SVM_ex(X_train, y_train)

for i in clsfic:
    print('When C is {} and gamma is {}, the score of prediction is: {:.6f}\
    '.format(i[0], i[1], i[2].score(X_test, y_test)))

# VIZmod.plot_ex(X_test, y_test, clsfic)
#
# grid= SVMmod.best_parameter(X_train, y_train)
# SVMmod.param_express(grid)
# VIZmod.plot_heatmap(grid)
# best_clf= SVMmod.SVM_one(X_train, y_train, grid=grid)
# # best_clf= SVMmod.SVM_one(X_train, y_train, C= 10, gamma= 10)
# # VIZmod.plot_onecase(X_test, y_test, grid, best_clf)
# VIZmod.plot_onecase(X_test, y_test, clf= best_clf)
