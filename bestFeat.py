from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def split_data(feat):
    dir = 'data/prep_featurizer/'
    X = pd.read_csv('%ssmote_train_%s_X.csv'%(dir,feat))
    y = pd.read_csv('%ssmote_train_%s_y.csv'%(dir,feat))
    X_train, X_test, y_train, y_test = train_test_split(X, y['failure'].ravel(), test_size=0.5, random_state=0)
    return X_train, X_test, y_train, y_test


for i in ['rfe','for','seq_fwd','seq_bwd']:
    print(i)
    X_train, X_val, y_train, y_val = split_data(i)
    svclassifier = SVC(kernel='poly', degree=6)
    svclassifier.fit(X_train, y_train)  
    y_pred = svclassifier.predict(X_val)
    print(confusion_matrix(y_val,y_pred))
    print(classification_report(y_val,y_pred))

    

