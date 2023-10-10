import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def knn(X_train, X_test, y_train):    
    KNNClassifier = KNeighborsClassifier(n_neighbors=20)
    KNNClassifier.fit(X_train, y_train)
    labels = KNNClassifier.predict(X_test)
    return labels
