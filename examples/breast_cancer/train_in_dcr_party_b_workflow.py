import os

from decentriq_deployment.decentriq_deployment import PartyB


if __name__ == "__main__":
    python_computation_filename = "examples/breast_cancer/training_script_for_decentriq.py"
    handler = PartyB(
                     credentials_file="credentials",
                     python_computation_filename=None,
                     data_clean_room_name=None
                    )

    # Read and remove DCR ID temp file.
    with open("tmp_dcr_id", "r") as file:
        dcr_id = file.read().rstrip()

    handler.party_b_requisitions(data_name="party_b",
                                 data_filename="examples/breast_cancer/data/data_party_b.csv",
                                 dcr_id=dcr_id
                                 )
    os.remove("tmp_dcr_id")
