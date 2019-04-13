from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np


def load_data():
    dir = 'data/time_rolled_5_days/balanced/prep_orig_data/'
    X_train = pd.read_csv('%strain_X.csv' % (dir))
    y_train = pd.read_csv('%strain_y.csv' % (dir))
    #X_val = pd.read_csv('%svalid_X.csv' % (dir))
    #y_val = pd.read_csv('%svalid_y.csv' % (dir))
    test_X = pd.read_csv('%stest_X.csv' % (dir))
    test_y = pd.read_csv('%stest_y.csv' % (dir))

    X_test_set=test_X
    y_test_set=test_y
    #X_train.drop(labels=['serial_number'],inplace=True,axis=1)
    X_test_set.drop(labels=['serial_number'], inplace=True,axis=1)
    y_train=np.array(y_train.values).reshape(-1,)
    y_test_set = np.array(y_test_set.values).reshape(-1,)
    return X_train, X_test_set, y_train, y_test_set

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

