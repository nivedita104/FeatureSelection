from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.metrics import classification_report
from bestModel import load_data, show_confusion_matrix
import sys


# In[]

def decision_tree(X_train, X_test, y_train, y_test, viz=False):
    tuning_parameters = {'min_samples_split': range(10, 500, 10),
                         'max_depth': range(1, 20, 2),
                         'max_features':range(1,X_train.shape[1])}
    clf = GridSearchCV(estimator=tree.DecisionTreeClassifier(), param_grid=tuning_parameters, n_jobs=-1, cv=7)
    clf.fit(X_train, y_train)
    print('******** Decision Tree *******')
    print('Best score for data:', clf.best_score_)
    print('Best value for min samples to split:', clf.best_estimator_.min_samples_split)
    print('Best max depth value:', clf.best_estimator_.max_depth)
    print('Best value to decide how many features to consider for splitting:', clf.best_estimator_.max_features)

    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    show_confusion_matrix(y_true, y_pred)
    if viz:
        from sklearn.externals.six import StringIO
        import pydotplus
        from IPython.display import Image
        from sklearn.tree import export_graphviz
        dot_data = StringIO()
        export_graphviz(clf.best_estimator_, out_file=dot_data,
                        class_names=["0","1"],
                        filled=True, rounded=True,
                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        Image(graph.create_png())

    return clf


X_train, X_test, y_train, y_test = load_data()
print(X_train.shape)

optimized_decision_tree = decision_tree(X_train, X_test, y_train, y_test)



# In[]

'''
Best score for data: 0.9880074620236298
Best value for min samples to split: 10
Best max depth value: 7
Best value to decide how many features to consider for splitting: 1
              precision    recall  f1-score   support

           0       0.99      1.00      0.99      4727
           1       0.81      0.30      0.43        98

   micro avg       0.98      0.98      0.98      4825
   macro avg       0.90      0.65      0.71      4825
weighted avg       0.98      0.98      0.98      4825

'''
