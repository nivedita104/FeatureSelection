from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def load_data():
    dir = 'data/prep_featurizer/'
    X = pd.read_csv('%ssmote_train_for_X.csv' % (dir))
    y = pd.read_csv('%ssmote_train_for_y.csv' % (dir))
    X_train, X_test, y_train, y_test = train_test_split(X, y['failure'].ravel(), test_size=0, random_state=0)
    test_X = pd.read_csv('%stest_for_X.csv' % (dir))
    test_y = pd.read_csv('%stest_y.csv' % (dir))
    return X_train, test_X, y_train, test_y


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


X_train, X_test, y_train, y_test = load_data()
print(X_train.shape)

SVM_CV(X_train, X_test, y_train, y_test)


'''
Best score for data1: 0.9870041039671683
Best C: 10
Best Kernel: linear
Best Gamma: auto_deprecated
              precision    recall  f1-score   support

           0       0.99      1.00      0.99     15791
           1       0.93      0.30      0.45       291

   micro avg       0.99      0.99      0.99     16082
   macro avg       0.96      0.65      0.72     16082
weighted avg       0.99      0.99      0.98     16082


'''