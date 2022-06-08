import pickle
import pandas as pd

from sklearn import svm
from sklearn.model_selection import GridSearchCV


def get_best_svm_model(x, y):
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    try:
        clf.fit(x, y)
    except ValueError as err:
        # Handle expected error during Decentriq computational tests.
        if str(err) == "Found array with 0 sample(s) (shape=(0,)) while a minimum of 1 is required.":
            print('validation passed')
        else:
            raise Exception(err)
    return clf


def get_best_svm_model_DCR(data_party_a, data_party_b):
    """
    Using the index, combine the data and get the best SVM model.
    For now getting the inner join of the two sets seems to be enough.
    """
    combined_data = pd.merge(data_party_a, data_party_b, left_index=True, right_index=True)
    # HACK: column names are not passed but we know that 'target' is the last column
    # x = combined_data.drop('target', axis=1)
    # y = combined_data['target']
    x = combined_data.iloc[:, :-1]
    y = combined_data.iloc[:, -1:]
    return get_best_svm_model(x, y)


if __name__ == "__main__":
    try:
        data_party_a = pd.read_csv("/input/party_a/dataset.csv", index_col=0)
        data_party_b = pd.read_csv("/input/party_b/dataset.csv", index_col=0)
    except pd.errors.EmptyDataError as err:
        # Handle expected error during Decentriq computational tests.
        if str(err) == 'No columns to parse from file':
            print('validation passed')
        else:
            raise Exception(err)

    model = get_best_svm_model_DCR(data_party_a, data_party_b)

    # Write to output file.
    with open('/output/model.pkl', 'wb') as file:
        pickle.dump(model, file)
