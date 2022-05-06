"""
NOTE: Was used to get the party A and party B data, but other that that this is an obsolete file.
      To be deleted.

DCR Scenario 1:
    We assume that our data comes from two parties, Party_A and Party_B.
    No party has access to the other's data, but a common identifier exists.
    We train an SVM using Grid Search and test using all features.
"""

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
    return x_train, x_test, y_train, y_test


def get_best_svm_model(x, y):
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(x, y)

    return clf.best_estimator_


def simulate_different_sources(x, y, ratio):
    """
    Split dataset column-wise into two halfs.
    Party_A data will contain a subset of the features.
    Party_B data will only contain the labels.
    """
    x["y"] = y
    data_party_a = x.iloc[:, :int(len(x.columns) * ratio)]
    data_party_b = x.iloc[:, -1]

    data_party_a = data_party_a.sample(frac=1)
    data_party_b = data_party_b.sample(frac=1)

    return data_party_a, data_party_b


def get_best_svm_model_DCR(data_party_a, data_party_b):
    """
    Using the index, combine the data and get the best SVM model.
    For now getting the inner join of the two sets seems to be enough.
    """
    combined_data = pd.merge(data_party_a, data_party_b, left_index=True, right_index=True)
    x = combined_data.drop('y', axis=1)
    y = combined_data['y']
    return get_best_svm_model(x, y)


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_data()

    ratio = 0.1  # very small values can result in a zero number of features
    data_party_a, data_party_b = simulate_different_sources(x_train, y_train, ratio=ratio)

    model = get_best_svm_model_DCR(data_party_a, data_party_b)
    x_test = x_test.iloc[:, :int(len(x_test.columns) * ratio)]  # use only half of the features

    preds = model.predict(x_test)

    acc = accuracy_score(preds, y_test)
    auc = roc_auc_score(preds, y_test)
    f1_score = f1_score(preds, y_test)
    print(f"Accuracy score on the test set: {acc}")
    print(f"AUC on the test set: {auc}")
    print(f"F1-Score on the test set: {f1_score}")

    # export data
    data_party_a.to_csv("data_party_a.csv")
    data_party_b.to_csv("data_party_b.csv")
