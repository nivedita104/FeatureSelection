from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def load_data():
    dir = 'data/prep_featurizer/'
    X = pd.read_csv('%ssmote_train_for_X.csv' % (dir))
    y = pd.read_csv('%ssmote_train_for_y.csv' % (dir))
    X_train, X_test, y_train, y_test = train_test_split(X, y['failure'].ravel(), test_size=0.3, random_state=0)
    test_X = pd.read_csv('%stest_for_X.csv' % (dir))
    test_y = pd.read_csv('%stest_y.csv' % (dir))
    return X_train, X_test, y_train, y_test

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

