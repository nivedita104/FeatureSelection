from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

def upsample(feat):
    dir = 'data/prep_featurizer/'
    X = pd.read_csv('%strain_%s_X.csv'%(dir,feat))
    y = pd.read_csv('%strain_y.csv'%(dir))
    print(y['failure'].value_counts())
    print("head")
    cols = list(X.head(0))
    col = ", ".join(str(x) for x in cols)
    smt = SMOTE(sampling_strategy = 0.6)#(sampling_strategy = 0.4, categorical_features=[1])
    X_train, y_train = smt.fit_sample(X, y['failure'].ravel())
    print(np.unique(y_train,return_counts=True))

    np.savetxt('%ssmote_train_%s_X.csv'%(dir,feat), X_train, delimiter=",",header=col,fmt="%i", comments='')
    np.savetxt('%ssmote_train_%s_y.csv'%(dir,feat),y_train, delimiter=",",header='failure',fmt="%i", comments='')


upsample('for')
upsample('seq_bwd')
upsample('seq_fwd')
upsample('rfe')

