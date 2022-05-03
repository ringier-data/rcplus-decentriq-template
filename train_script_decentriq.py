import pickle
import pandas as pd

from sklearn import svm
from sklearn.model_selection import GridSearchCV

def get_best_svm_model(x, y):
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    clf.fit(x, y)
    return clf.best_estimator_


def get_best_svm_model_DCR(data_party_a, data_party_b):
    """
    Using the index, combine the data and get the best SVM model.
    For now getting the inner join of the two sets seems to be enough.
    """
    combined_data = pd.merge(data_party_a, data_party_b, left_index=True, right_index=True)
    x = combined_data.drop('y', axis=1)
    y = combined_data['y']
    return get_best_svm_model(x, y)
    #return {'x': x, 'y':y}


if __name__ == "__main__":
    try:
        data_party_a = pd.read_csv("/input/party_a/dataset.csv", skiprows=1,
                                   names=["id", "mean radius", "mean texture", "mean perimeter"])
        data_party_b = pd.read_csv("/input/party_b/dataset.csv",skiprows=1,
                                   names=["id", "y"])

        data_party_a.set_index("id", inplace=True)
        data_party_b.set_index("id", inplace=True)

        model = get_best_svm_model_DCR(data_party_a, data_party_b)

        # Write to output file.
        with open('/output/model.pkl', 'wb') as file:
            pickle.dump(model, file)

    # Handle expected errors
    except pd.errors.EmptyDataError as err:
        if str(err) == 'No columns to parse from file':
            print('validation passed')
        else:
            raise Exception(err)
    except ValueError as err:
        if str(err) == "Found array with 0 sample(s) (shape=(0,)) while a minimum of 1 is required.":
            print('validation passed')
        else:
            raise Exception(err)
