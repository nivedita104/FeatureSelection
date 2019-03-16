from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from bestModel import load_data, show_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix

def SVM_CV(X_train, X_test, y_train, y_test):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    clf = GridSearchCV(estimator=SVC(), param_grid=tuned_parameters, n_jobs=-1, cv=7)
    clf.fit(X_train, y_train)
    print('Best score for data1:', clf.best_score_)
    print('Best C:', clf.best_estimator_.C)
    print('Best Kernel:', clf.best_estimator_.kernel)
    print('Best Gamma:', clf.best_estimator_.gamma)
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    show_confusion_matrix(y_true, y_pred)

X_train, X_test, y_train, y_test = load_data()
print(X_train.shape)

SVM_CV(X_train, X_test, y_train, y_test)


'''
Best score for data1: 0.9880074620236298
Best C: 10
Best Kernel: linear
Best Gamma: auto_deprecated
              precision    recall  f1-score   support

           0       0.99      1.00      0.99      4727
           1       0.87      0.28      0.42        98

   micro avg       0.98      0.98      0.98      4825
   macro avg       0.93      0.64      0.71      4825
weighted avg       0.98      0.98      0.98      4825


'''