import datetime

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
        self.publish_data_clean_room()
        self.upload_data(data_name, data_filename)

    def execute_computations(self, extraction_folder="."):
        return super().execute_computations(extraction_folder)


if __name__ == "__main__":
    python_computation_filename = "examples/breast_cancer/training_script_for_decentriq.py"
    handler = PartyA(
                     credentials_file="credentials",
                     python_computation_filename=python_computation_filename,
                     data_clean_room_name=f"ExampleBreastCancer_{datetime.date.today()}"
                    )
    handler.party_a_requisitions(data_name="party_a", data_filename="examples/breast_cancer/data/data_party_a.csv")

    # Save DCR ID to temp file for Party B to load from.
    with open("tmp_dcr_id", "w") as file:
        file.write(handler.python_dcr_id)

    # TODO: auto-check if party B has uploaded data or do the computation in another file (overkill for showcase?)
    input("Press any key when party B data are uploaded")

    handler.execute_computations()
