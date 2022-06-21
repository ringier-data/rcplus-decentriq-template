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


if __name__ == "__main__":
    python_computation_filename = "examples/user_vectors/training_script_for_decentriq.py"
    handler = PartyA(
                     credentials_file="credentials",
                     python_computation_filename=python_computation_filename,
                     data_clean_room_name=f"ExampleUserVectors_{datetime.date.today()}"
                    )

    # Read DCR ID temp file.
    with open("tmp_dcr_id", "r") as file:
        dcr_id = file.read().rstrip()

    # Execute the python computation in the DCR.
    handler.python_dcr_id = dcr_id
    handler.initialize_session(handler.credentials_file)
    handler.execute_computations(extraction_folder="examples/user_vectors")
    os.remove("tmp_dcr_id")
