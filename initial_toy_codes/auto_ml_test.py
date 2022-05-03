import sklearn
import pandas as pd
import autosklearn.classification

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


def main():
    data = datasets.load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df.target = 1 - df.target
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=4)
    
    
    automl = autosklearn.classification.AutoSklearnClassifier(n_jobs=4)
    automl.fit(X_train, y_train)
    
    predictions = automl.predict(X_test)
    print(f"Accuracy score : {accuracy_score(y_test, predictions): 6.3f}")
    print(f"Balanced accuracy score : {roc_auc_score(y_test, predictions): 6.3f}")

if __name__ == '__main__':
    main()
