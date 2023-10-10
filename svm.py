from sklearn.svm import SVC

def svm(X_train, X_test, y_train):
    model = SVC(kernel='poly')
    model.fit(X_train, y_train)
    labels = model.predict(X_test)
    return labels