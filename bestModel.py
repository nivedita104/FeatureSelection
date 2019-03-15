from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def load_data():
    dir = 'data/prep_featurizer/'
    X = pd.read_csv('%ssmote_train_for_X.csv'%(dir))
    y = pd.read_csv('%ssmote_train_for_y.csv'%(dir))
    X_train, X_test, y_train, y_test = train_test_split(X, y['failure'].ravel(), test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test


def SVM_CV(X_train, X_test, y_train, y_test):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    clf = GridSearchCV(estimator=SVC(), param_grid=tuned_parameters, n_jobs=-1,cv=7)
    clf.fit(X_train, y_train)
    print('Best score for data1:', clf.best_score_)
    print('Best C:',clf.best_estimator_.C) 
    print('Best Kernel:',clf.best_estimator_.kernel)
    print('Best Gamma:',clf.best_estimator_.gamma)
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

X_train, X_val, y_train, y_val = load_data()
print(X_train.shape)
SVM_CV(X_train, X_val, y_train, y_val)
