import datetime
import pandas as pd
import hashlib

from decentriq_deployment.decentriq_deployment import DecentriqDeployment


class PartyA(DecentriqDeployment):
    """
    Party A will be responsible for publishing the DCR, defining the python computation and
    uploading the dataset features.
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

    def party_a_requisitions(self, data_name, data_filename):
        self.initialize_session(self.credentials_file)
        self.publish_data_clean_room(second_party_user_email=self.user_email)  # NOTE: same as party A for demo purposes
        self.upload_data(data_name, data_filename)


def read_schema(filename):
    """
    Args:
        filename: path to a csv file with each row corresponding to a column's name
                  and data type
    Returns:
        schema: The returned format will be a list of tuples (col_name, col_datatype),
                as required by Decentriq.
    """
    df = pd.read_csv(filename)
    schema = [(row[0], row[1]) for row in df.values]
    return schema


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
    # Read schemas from corresponding files
    # NOTE: In practice, party B must share it's data schema to the DCR publisher (party A).
    schema1 = read_schema("examples/user_vectors/data/party_a_schema.csv")
    schema2 = read_schema("examples/user_vectors/data/party_b_schema.csv")

    # Apply hashing function before uploading to DCR.
    print("Hashing the user emails...")
    create_hashed_dataset(
                          dataset_filename="examples/user_vectors/data/party_a_user_vectors.csv",
                          email_mapping_table="examples/user_vectors/data/party_a_map.csv",
                          hashing_function=lambda x: hashlib.sha224(x.encode()).hexdigest(),
                          output_filename="examples/user_vectors/data/party_a_user_vectors_hashed.csv"
                         )
    print("Hashing finished.")

    python_computation_filename = "examples/user_vectors/training_script_for_decentriq.py"
    handler = PartyA(
                     credentials_file="credentials",
                     python_computation_filename=python_computation_filename,
                     data_clean_room_name=f"ExampleUserVectors_{datetime.date.today()}",
                     schema1=schema1,
                     schema2=schema2
                    )

    handler.party_a_requisitions(
                                 data_name="party_a",
                                 data_filename="examples/user_vectors/data/party_a_user_vectors_hashed.csv"
                                )

    # Save DCR ID to a temp file for Party B to load from (and also to load when executing computations).
    with open("tmp_dcr_id", "w") as file:
        file.write(handler.python_dcr_id)
    # Save Training Node ID for executing computations at a latter stage.
    with open("tmp_training_node_id", "w") as file:
        file.write(handler.training_node_id)
