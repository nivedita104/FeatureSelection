from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from bestModel import load_data, show_confusion_matrix
import sys


# In[]

def logistic_reg(X_train, X_test, y_train, y_test):
    tuning_parameters = {'penalty': ["l2"],
                         'class_weight': ['balanced'],
                         'random_state': [42],
                         'tol': [1e-3, 1e-4],
                         'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                         'C': [0.1, 0.5, 1, 2, 10, 100, 1000]}
    clf = GridSearchCV(estimator=LogisticRegression(), param_grid=tuning_parameters, n_jobs=-1, cv=7)
    clf.fit(X_train, y_train)
    print('******** Logistic regression ******')
    print('Best score for data:', clf.best_score_)
    print('Best tolerance value:', clf.best_estimator_.tol)
    print('Best solver:', clf.best_estimator_.solver)
    print('Best value for C', clf.best_estimator_.C)

    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    show_confusion_matrix(y_true, y_pred)

    return clf


X_train, X_test, y_train, y_test = load_data()
print(X_train.shape)

best_log_regression = logistic_reg(X_train, X_test, y_train, y_test)

# In[]

'''
******** Logistic regression ******
Best score for data: 0.9563826952118681
Best tolerance value: 0.001
Best solver: newton-cg
Best value for C 0.5

              precision    recall  f1-score   support

           0       0.99      0.96      0.98      4727
           1       0.24      0.61      0.35        98

   micro avg       0.95      0.95      0.95      4825
   macro avg       0.62      0.79      0.66      4825
weighted avg       0.98      0.95      0.96      4825



'''

''' TIME ROLLED 5 days
******** Logistic regression ******
Best score for data: 0.9286436066848037
Best tolerance value: 0.001
Best solver: newton-cg
Best value for C 100
              precision    recall  f1-score   support

           0       0.99      0.94      0.96      3152
           1       0.18      0.68      0.28        65

   micro avg       0.93      0.93      0.93      3217
   macro avg       0.59      0.81      0.62      3217
weighted avg       0.98      0.93      0.95      3217

'''