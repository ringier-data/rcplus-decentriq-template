# Decentriq Template

Templating package for the training of machine learning models through the Decetriq confidential computing platform.
The scope is the quick setup and publishing of Data Clean Rooms (DCRs), the data uploading from multiple parties, and the execution of computations through the abstraction of the Decentriq Python SDK's boilerplate code.


## Current scenarios and test cases

- Scenario 1: Party_A provides features, Party_B provides features and labels. Data from both parties contain a shared unique identifier. The model is trained with features from both parties. Party_A gets the model and evaluates its performance with the test set that contains all features.
- Scenario 2: Party_A provides features and Party_B provides only labels. We match features to labels using a shared unique identifier and train the model.
- Scenario 3: Party_A provides the model/training code and Party_B the features and labels. Party_A gets only the model. Can we provide predictions without knowing anything about Party_B's features? (or at least knowing only a mean?) If we use a mean for the missing features in the test set, the (working/not-working) results become heavily dataset-specific.
- Scenario 4: Party A tries to predict party B's features. Can be broken down be a subset of scenario 2.

## Examples
### Train on breast cancer scikit-learn dataset
- Implements scenario 2
- PartyA and PartyB classes inherit from DecentriqDeployment
- PartyA defines and publishes the DCR, uploads data (features) and waits for PartyB to upload its data
- PartyB uploads data (labels)
- PartyA executes the computation and get the trained model

To execute:
1. run `train_in_dcr_party_a_workflow.py` and leave the script running without pressing `enter`
1. run `train_in_dcr_party_b_workflow.py`
1. press `enter` in the first script

Credentials (email and API token) must be stored in `credentials` file in repo root.

TODO: We can avoid the above quick-and-dirty process by creating a third script decicated to execution of the computation (if not deemed overkill).


## Questions and open issues
**Q:** How are packages installed in decentriq? Do we have restrictions regarding sklearn version, etc.? \
**A:** We can't install anything. Decentriq data clean rooms contain some pre-installed libraries.

**Q:** Decentriq offers Scikit-Learn installed, but not auto-sklearn, which is a different package. Can we use auto-sklearn from the vanilla/pure sklearn package? \
**A:** (outdated) No, best approach for now seems to be using some ML models from sklearn and a gridsearch.\
**A:** Decentriq might be able to provide as with a debug VM (with limited security quarantees and non-granted availability) that contains the required packages.

**Q:** Use different credentials\
**A**: found how-to; todo: implement

**Issue:** Decentriq is facing issues with loading abstract files (instread of tabular data) in the DCRs. Specifically, it is only possible to use one (and only one) file that is unstructured. Using more than one is for the moment disabled, as some edge cases were found that are not easy to control in confidential computing yet. Decentriq is currently enabling them. Decentriq suggests that reasons to use the tables rather than files when possible is that they are much easier to control, there is data validation in place for them and are optimized in the read-write process.
