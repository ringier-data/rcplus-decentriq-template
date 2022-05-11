import datetime
from decentriq_deployment.decentriq_deployment import DecentriqDeployment


if __name__ == "__main__":
    python_computation_filename = "examples/breast_cancer/training_script_for_decentriq.py"
    handler = DecentriqDeployment(
                                  credentials_file="credentials",
                                  python_computation_filename=python_computation_filename,
                                  data_clean_room_name=f"ExampleBreastCancer_{datetime.date.today()}"
    )
    handler.deploy_workflow()
