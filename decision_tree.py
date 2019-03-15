from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import sys


# In[]

def load_data():
    dir = 'feat_svm/data/prep_featurizer/'
    X = pd.read_csv('%ssmote_train_for_X.csv' % (dir))
    y = pd.read_csv('%ssmote_train_for_y.csv' % (dir))
    X_train, X_test, y_train, y_test = train_test_split(X, y['failure'].ravel(), test_size=0, random_state=0)
    test_X = pd.read_csv('%stest_for_X.csv' % (dir))
    test_y = pd.read_csv('%stest_y.csv' % (dir))
    return X_train, test_X, y_train, test_y


def show_confusion_matrix(validations, predictions):

    matrix = confusion_matrix(validations, predictions)
    plt.figure(figsize=(10, 10))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=["0","1"],
                yticklabels=["0","1"],
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def decision_tree(X_train, X_test, y_train, y_test, viz=False):
    tuning_parameters = {'min_samples_split': range(10, 500, 10),
                         'max_depth': range(1, 20, 2),
                         'max_features':range(1,X_train.shape[1])}
    clf = GridSearchCV(estimator=tree.DecisionTreeClassifier(), param_grid=tuning_parameters, n_jobs=-1, cv=7)
    clf.fit(X_train, y_train)
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
Best score for data: 0.9880611864195995
Best value for min samples to split: 10
Best max depth value: 17
Best value to decide how many features to consider for splitting: 6

              precision    recall  f1-score   support
           0       0.99      1.00      0.99     15791
           1       0.66      0.48      0.55       291
           
   micro avg       0.99      0.99      0.99     16082
   macro avg       0.83      0.74      0.77     16082
weighted avg       0.98      0.99      0.99     16082
'''
