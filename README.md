# Decentriq Template

Templating package for the training of machine learning models through the Decetriq confidential computing platform.
The scope is the quick setup and publishing of Data Clean Rooms (DCRs), the data uploading from multiple parties, and the execution of computations through the abstraction of the Decentriq Python SDK's boilerplate code.


## Current scenarios and test cases

- Scenario 1: Party_A provides features, Party_B provides features and labels. Data from both parties contain a shared unique identifier. The model is trained with features from both parties. Party_A gets the model and evaluates its performance with the test set that contains all features.
- Scenario 2: Party_A provides features and Party_B provides only labels. We match features to labels using a shared unique identifier and train the model.
- Scenario 3: Party A tries to predict party B's features. Can be broken down be a subset of scenario 2.

## Examples
### Train on breast cancer scikit-learn dataset
- Implements scenario 2
- PartyA and PartyB classes inherit from DecentriqDeployment
- PartyA defines and publishes the DCR, uploads data (features) and waits for PartyB to upload its data
- PartyB uploads data (labels)
- PartyA executes the computation and gets the trained model

To execute:
1. run `train_in_dcr_party_a_workflow_1_upload.py`
1. run `train_in_dcr_party_b_workflow.py`
1. run `train_in_dcr_party_a_workflow_2_execute.py`


Credentials (email and API token) must be stored in `credentials` file in repo root.


## Questions and open issues
**Q:** How are packages installed in decentriq? Do we have restrictions regarding sklearn version, etc.? \
**A:** Decentriq data clean rooms do not allow installing packages, but instead come with some pre-installed libraries.


**Issue:** There are some issues regarding the usage of abstract files (instread of tabular data) in the DCRs. Specifically, it is only possible to use one (and only one) file that is unstructured. Using more than one is for the moment disabled, as some edge cases were found that are not easy to control in confidential computing yet. Decentriq is currently enabling them. Suggested reasons for using tabular rather than abstract files (when possible) is that they are easier to control, there is data validation in place for them and are optimized in the read-write process.
