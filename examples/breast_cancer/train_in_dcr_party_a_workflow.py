import datetime

from decentriq_deployment.decentriq_deployment import PartyA


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

    # TODO: auto-check if party B has uploaded data or reload hanlder in another file (overkill for showcase?)
    input("Press any key when party B data are uploaded")

    handler.execute_computations()
