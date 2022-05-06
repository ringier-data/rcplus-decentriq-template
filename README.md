# Decentriq Template

Templating package for the training of machine learning models through the Decetriq confidential computing platform.
The scope is the quick setup and publishing of the Data Clean Rooms (DCRs), as well as the data uploading and execution
of computations, through the abstraction of the Decentriq Python SDK's boilerplate code.


## Current scenarios and test cases

- Scenario 1: Party_A provides features, Party_B provides features and labels. Features from both parties contain a common unique identifier. The model is trained with features from both parties. Party_A gets the model and evaluates its performance with the test set that contains all features.
- Scenario 2: Party_A provides features, Party_B provides only labels. We match features to labels using a common unique identifier. The model is trained with features from party_A. Party_A gets the model and evaluates its performance with the test set that contains the subset of features.
- Scenario 3: Party_A provides the model and Party_B the features and labels. Party_A gets only the model. Can we provide predictions without knowing anything about Party_B's features? (or at least knowing only a mean?) If we use a mean for the missing features in the test set, the (working/not-working) results become heavily dataset-specific.

## Examples
### Train on breast cancer scikit-learn dataset
- implements scenario 2
- PartyA and PartyB classes inherit from DecentriqDeployment
- PartyA defines and publishes the DCR, uploads data (features) and waits for PartyB to upload its data
- PartyB uploads data (labels)
- PartyA executes the computation and get the trained model

To execute:
1. run `train_in_dcr_party_a_workflow.py` and leave the script running without pressing `enter`
1. run `train_in_dcr_party_b_workflow.py`
1. press `enter` in the first script

Crentials (email and API token) must be stored in a `credentials` files in repo root.

TODO: We can avoid the above quick-and-dirty process by creating a third script decicated to execution of the computation (if not deemed overkill).


## Questions and open issues
Q: How are packages installed in decentriq? Do we have restrictions regarding sklearn version, etc.? \
A: We can't install anything. Decentriq data clean rooms contain some pre-installed libraries.

Q: Decentriq offers Scikit-Learn installed, but not auto-sklearn, which is a different package. Can we use auto-sklearn from the vanilla/pure sklearn package? \
A: (outdated) No, best approach for now seems to be using some ML models from sklearn and a gridsearch.\
A: Decentriq might be able to provide as with a debug VM (with limited security quarantees and non-granted availability) that contains the required packages.

Q: Use different credentials\
A: found how-to; todo: implement