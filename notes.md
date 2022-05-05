### Questions and open issues
Q: How are packages installed in decentriq? Do we have restrictions regarding sklearn version, etc.? \
A: We can't install anything. Decentriq data clean rooms contain some pre-installed libraries.

Q: Decentriq offers Scikit-Learn installed, but not auto-sklearn, which is a different package. Can we use auto-sklearn from the vanilla/pure sklearn package? \
A: (outdated) No, best approach for now seems to be using some ML models from sklearn and a gridsearch.
A: Decentriq might be able to provide as with a debug VM (with limited security quarantees and non-granted availability) that contains the required packages.

### Auto-sklearn
`pip install auto-sklearn`

Installation of sklearn-auto may need a few more things kuje SWIG, c++11.
https://automl.github.io/auto-sklearn/master/installation.html#installation
For now, installation of this package is not our concern.

### Load toy data and split them to set_a and set_b
We can use the breast_cancer dataset to simulate information.
TODO: update

### Test cases

- Scenario 1: Party_A provides features, Party_B provides features and labels. Features from both parties contain a common unique identifier. The model is trained with features from both parties. Party_A gets the model and evaluates its performance with the test set that contains all features.
- Scenario 2: Party_A provides features, Party_B provides only labels. We match features to labels using a common unique identifier. The model is trained with features from party_A. Party_A gets the model and evaluates its performance with the test set that contains the subset of features.
- Scenario 3: Party_A provides the model and Party_B the features and labels. Party_A gets only the model. Can we provide predictions without knowing anything about Party_B's features? (or at least knowing only a mean?) If we use a mean for the missing features in the test set, the (working/not-working) results become heavily dataset-specific.

### Update: Runned a script in a DCR web API, TODO: fill details


### Update: Runned a script in a DCR Python API, TODO: fill details