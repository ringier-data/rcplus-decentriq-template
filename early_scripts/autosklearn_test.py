import time
import pickle
import pandas as pd
import autosklearn.classification

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


def test_auto_sklearn():
    """
    A test with auto sklearn and the dataset as-is. The library handles all the data and feature
    preprocessing load (e.g. one hot encoding, dimensionality reduction, etc.), we only need to
    give the raw input.

    NOTE: The default memory limit (3072) returns a MEMOUT error, increasing the value to 10 GBs.
          The manual suggest that a good total time limit is 1 day with a time limit of 30 minutes
          for a single run.
          https://automl.github.io/auto-sklearn/master/manual.html

    NOTE: we do get some exceptions when the script is finished, however these are automatically
          ignored, since they concern the deletion of tmp files by forked subprocesses. Hopefully
          the DCR won't complain about these, and we will still the downloaded trained ensemble model.
          The error may be be due to the fact that we are logged as root in the VMs.
    """
    data = datasets.load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df.target = 1 - df.target
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=4)

    # Fit with auto-sklearn.
    start = time.time()
    automl = autosklearn.classification.AutoSklearnClassifier(n_jobs=8, memory_limit=7000)
    automl.fit(X_train, y_train)

    # Predict
    predictions = automl.predict(X_test)
    print(f"Accuracy score : {accuracy_score(y_test, predictions): 6.3f}")
    print(f"Balanced accuracy score : {roc_auc_score(y_test, predictions): 6.3f}")
    end = time.time()
    print(f"Time passed in seconds {end - start}")

    # save model
    with open("early_scripts/test_automl_model.pkl", "wb") as file:
        pickle.dump(automl, file)


if __name__ == '__main__':
    test_auto_sklearn()
