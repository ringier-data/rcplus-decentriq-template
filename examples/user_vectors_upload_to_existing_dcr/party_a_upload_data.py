import hashlib
import pandas as pd

from decentriq_deployment.decentriq_deployment import DecentriqDeployment


class PartyAUpload(DecentriqDeployment):
    """
    Upload data to an already existing DCR.
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

    def party_a_requisitions(self, data_name, data_filename, dcr_id):
        self.python_dcr_id = dcr_id
        self.initialize_session(self.credentials_file)
        self.upload_data(data_name, data_filename)


def create_hashed_dataset(dataset_filename, email_mapping_table, hashing_function, output_filename):
    """
    Map IDs to mails and then hash/encrypt the emails as a common identifier to be used in the DCR.

    Args:
        dataset_filename:
        email_mapping_table:
        hashing_function: It would be better to keep this private for real applications.
        output_filename:
    Returns:
        None
    """
    dataset = pd.read_csv(dataset_filename)
    mapping = pd.read_csv(email_mapping_table)
    mapping = {row[0]: row[1] for row in mapping.values}
    dataset["user_id"] = [mapping[id] for id in dataset["user_id"].values]
    dataset["user_id"] = dataset["user_id"].apply(hashing_function)
    dataset.to_csv(output_filename, index=False)


if __name__ == "__main__":

    # Apply hashing function before uploading to DCR.
    print("Hashing the user emails...")
    create_hashed_dataset(
                          dataset_filename="examples/user_vectors_upload_to_existing_dcr/data/party_a_user_vectors.csv",
                          email_mapping_table="examples/user_vectors_upload_to_existing_dcr/data/party_a_map.csv",
                          hashing_function=lambda x: hashlib.sha224(x.encode()).hexdigest(),
                          output_filename="examples/user_vectors_upload_to_existing_dcr/data/party_a_user_vectors_hashed.csv"
                         )
    print("Hashing finished.")

    handler = PartyAUpload(
                           credentials_file="credentials",
                           python_computation_filename=None,
                           data_clean_room_name=None,
                          )

    with open("tmp_dcr_id", "r") as file:
        dcr_id = file.read().rstrip()

    handler.party_a_requisitions(data_name="party_a",
                                 data_filename="examples/user_vectors_upload_to_existing_dcr/data/party_a_user_vectors_hashed.csv",
                                 dcr_id=dcr_id
                                 )
