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


with open("model.pkl", "rb") as file:  # TODO: add a parameterized model name
    model = pickle.load(file)

x_train, x_test, y_train, y_test = get_data()

preds = model.predict(x_test[['mean radius', 'mean texture', 'mean perimeter']])
acc = accuracy_score(preds, y_test)
auc = roc_auc_score(preds, y_test)
f1 = f1_score(preds, y_test)
print(f"Accuracy score on the test set: {acc}")
print(f"AUC on the test set: {auc}")
print(f"F1-Score on the test set: {f1}")

rnd_pred = np.random.randint(0, 2, len(y_test))
acc_rnd = accuracy_score(preds, rnd_pred)
auc_rnd = roc_auc_score(preds, rnd_pred)
f1_score_rnd = f1_score(preds, rnd_pred)
print(f"Random Accuracy score on the test set: {acc_rnd}")
print(f"Random AUC on the test set: {auc_rnd}")
print(f"Random F1-Score on the test set: {f1_score_rnd}")
