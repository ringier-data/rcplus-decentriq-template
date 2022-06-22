import hashlib

from decentriq_deployment.decentriq_deployment import DecentriqDeployment
# TODO: move this to decentriq_deployment.utils?
from examples.user_vectors.train_in_dcr_party_a_workflow_1_upload import create_hashed_dataset


class PartyB(DecentriqDeployment):
    """
    Party B will be responsible only for uploading the dataset labels.
    """
    def __init__(
                 self,
                 credentials_file,
                 python_computation_filename,
                 data_clean_room_name,
                 schema1=None,
                 schema2=None
                ):
        super().__init__(
                         credentials_file,
                         python_computation_filename,
                         data_clean_room_name,
                         schema1,
                         schema2
                        )

    def party_b_requisitions(self, data_name, data_filename, dcr_id):
        self.python_dcr_id = dcr_id
        self.initialize_session(self.credentials_file)
        self.upload_data(data_name, data_filename)


if __name__ == "__main__":

    # Apply hashing function before uploading to DCR.
    print("Hashing the user emails...")
    create_hashed_dataset(
                          dataset_filename="examples/user_vectors/data/party_b_targets.csv",
                          email_mapping_table="examples/user_vectors/data/party_b_map.csv",
                          hashing_function=lambda x: hashlib.sha224(x.encode()).hexdigest(),
                          output_filename="examples/user_vectors/data/party_b_targets_hashed.csv"
                         )
    print("Hashing finished.")

    handler = PartyB(
                     credentials_file="credentials",
                     python_computation_filename=None,
                     data_clean_room_name=None
                    )

    # Read DCR ID temp file.
    with open("tmp_dcr_id", "r") as file:
        dcr_id = file.read().rstrip()

    # Upload Party B data
    handler.party_b_requisitions(data_name="party_b",
                                 data_filename="examples/user_vectors/data/party_b_targets_hashed.csv",
                                 dcr_id=dcr_id
                                 )
