import pickle
import pandas as pd
import os
import autosklearn.classification

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


def get_best_model(x, y):
    automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=300, n_jobs=8, memory_limit=8000)
    try:
        automl.fit(x, y)
    except ValueError as err:
        # Handle expected error during Decentriq computational tests.
        if str(err) == "Found array with 0 sample(s) (shape=(0, 1)) while a minimum of 1 is required.":
            print('validation passed')
        else:
            print(str(err))
            os._exit(1)
    return automl


def get_best_model_DCR(data_party_a, data_party_b):
    """
    Using the index, combine the data and get the best SVM model.
    For now getting the inner join of the two sets seems to be enough.
    """
    combined_data = pd.merge(data_party_a, data_party_b, left_index=True, right_index=True)
    x = combined_data.drop('y', axis=1)
    y = combined_data['y']
    return get_best_model(x, y)


if __name__ == "__main__":
    try:
        data_party_a = pd.read_csv("/input/party_a/dataset.csv", skiprows=1,
                                   names=["id", "mean radius", "mean texture", "mean perimeter"])
        data_party_b = pd.read_csv("/input/party_b/dataset.csv", skiprows=1,
                                   names=["id", "y"])
    except pd.errors.EmptyDataError as err:
        # Handle expected error during Decentriq computational tests.
        if str(err) == 'No columns to parse from file':
            print('validation passed')
        else:
            raise Exception(err)

    data_party_a.set_index("id", inplace=True)
    data_party_b.set_index("id", inplace=True)

    model = get_best_model_DCR(data_party_a, data_party_b)

    # Write to output file.
    with open('/output/model.pkl', 'wb') as file:
        pickle.dump(model, file)

    os._exit(0)
