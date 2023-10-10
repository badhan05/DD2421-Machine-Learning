import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

def cleanData(data):                
    data = data[data.y.notna()]    
    unique_labels_y = pd.unique(data.y)
    data = data[data.y != unique_labels_y[4]]
    data = data[data.y != unique_labels_y[5]]
    
    threshold = 10**3
    data = data[data.x1 < threshold]
    data = data[data.x1 > -threshold]       

    data.drop('x2', inplace=True, axis=1)
    data.drop('x3', inplace=True, axis=1)
    data.drop('x4', inplace=True, axis=1)
    data.drop('x5', inplace=True, axis=1)
    data.drop('x6', inplace=True, axis=1)
    data.drop('x12', inplace=True, axis=1)
    
    rows, cols = data.shape

    normalizer = StandardScaler()
    X = normalizer.fit_transform(data.iloc[:, 1:cols])
    y = data['y'].to_numpy()    
    return X, y

def get_evaluation_data():
    data = getData('EvaluateOnMe.csv', 14)
    data.drop('x2', inplace=True, axis=1)
    data.drop('x3', inplace=True, axis=1)
    data.drop('x4', inplace=True, axis=1)
    data.drop('x5', inplace=True, axis=1)
    data.drop('x6', inplace=True, axis=1)
    data.drop('x12', inplace=True, axis=1)    
    normalizer = StandardScaler()    
    X = normalizer.fit_transform(data)
    return X

def select_features(X, y):
    X_new = SelectKBest(f_classif, k=6).fit_transform(X, y)
    print(X_new.shape)
    return X_new

def getData(filename, cols=14):            
    return pd.read_csv(filename, usecols=range(1,cols))
