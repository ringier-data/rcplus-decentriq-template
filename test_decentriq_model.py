import pickle
import sklearn
import numpy as np

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from starting_train_script.svm_grid_search_DCR_2 import get_data

with open("model_from_decentric_with_python_sdk.pkl", "rb") as file:
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