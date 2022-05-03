import numpy as np
import pandas as pd

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def get_data():
    data = datasets.load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df.target = 1 - df.target
    X = df.drop('target', axis=1)
    y = df['target']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=4)
    return  x_train, x_test, y_train, y_test


def get_best_svm_model(x, y):
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(x, y)

    return clf.best_estimator_


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_data()
    x_train = x_train.iloc[:, :15]
    x_test = x_test.iloc[:, :15]

    model = get_best_svm_model(x_train, y_train)
    preds = model.predict(x_test)

    acc = accuracy_score(preds, y_test)
    auc = roc_auc_score(preds, y_test)
    f1_score = f1_score(preds, y_test)
    print(f"Accuracy score on the test set: {acc}")
    print(f"AUC on the test set: {auc}")
    print(f"F1-Score on the test set: {f1_score}")
    print(f"labels in the testing set{np.unique(y_test, return_counts=True)}")
