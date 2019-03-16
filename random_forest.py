from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from bestModel import load_data, show_confusion_matrix
import sys


# In[]

def decision_tree(X_train, X_test, y_train, y_test, viz=False):
    tuning_parameters = {'criterion':['gini','entropy'],
                         'n_estimators': range(10, 30, 5),
                         'min_samples_split': range(10, 500, 10),
                         'max_depth': range(1, 20, 2),
                         'max_features':range(1,X_train.shape[1])}
    clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=tuning_parameters, n_jobs=-1, cv=7)
    clf.fit(X_train, y_train)
    print('******** Random Forest *******')
    print('Best score for data:', clf.best_score_)
    print('Best no. of estimators:', clf.best_estimator_.n_estimators)
    print('Best criterion for splitting:', clf.best_estimator_.criterion)
    print('Best value for min samples to split:', clf.best_estimator_.min_samples_split)
    print('Best max depth value:', clf.best_estimator_.max_depth)
    print('Best value to decide how many features to consider for splitting:', clf.best_estimator_.max_features)

    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    show_confusion_matrix(y_true, y_pred)
    return clf


X_train, X_test, y_train, y_test = load_data()
print(X_train.shape)

optimized_random_forest = decision_tree(X_train, X_test, y_train, y_test)



# In[]

'''

'''
