import os
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


if __name__ == "__main__":
    python_computation_filename = "examples/breast_cancer/training_script_for_decentriq.py"
    handler = PartyA(
                     credentials_file="credentials",
                     python_computation_filename=python_computation_filename,
                     data_clean_room_name=f"ExampleBreastCancer_{datetime.date.today()}"
                    )

    # Read DCR ID temp file.
    with open("tmp_dcr_id", "r") as file:
        dcr_id = file.read().rstrip()

    # Execute the python computation in the DCR.
    handler.python_dcr_id = dcr_id
    handler.initialize_session(handler.credentials_file)
    handler.execute_computations(extraction_folder="examples/breast_cancer")
    os.remove("tmp_dcr_id")
