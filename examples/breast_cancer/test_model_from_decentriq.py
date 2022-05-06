import pickle
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split


def get_data():
    data = datasets.load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df.target = 1 - df.target
    X = df.drop('target', axis=1)
    y = df['target']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=4)
    return x_train, x_test, y_train, y_test


def print_metrics(preds, y):
    acc = accuracy_score(preds, y)
    auc = roc_auc_score(preds, y)
    f1 = f1_score(preds, y)
    print(f"Accuracy score on the test set: {acc}")
    print(f"AUC on the test set: {auc}")
    print(f"F1-Score on the test set: {f1}")


if __name__ == "__main__":
    with open("model.pkl", "rb") as file:  # TODO: add a parameterized model name
        model = pickle.load(file)

    x_train, x_test, y_train, y_test = get_data()

    preds = model.predict(x_test[['mean radius', 'mean texture', 'mean perimeter']])
    print_metrics(preds, y_test)

    rnd_pred = np.random.randint(0, 2, len(y_test))
    print_metrics(rnd_pred, y_test)