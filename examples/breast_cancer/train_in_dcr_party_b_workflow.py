from decentriq_deployment.decentriq_deployment import DecentriqDeployment


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
    python_computation_filename = "examples/breast_cancer/training_script_for_decentriq.py"
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
                                 data_filename="examples/breast_cancer/data/data_party_b.csv",
                                 dcr_id=dcr_id
                                 )
