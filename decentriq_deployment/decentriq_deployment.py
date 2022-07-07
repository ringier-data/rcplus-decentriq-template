import io
import decentriq_platform as dq
import decentriq_platform.sql as dqsql
import decentriq_platform.container as dqc

from decentriq_platform.container.proto import MountPoint
from decentriq_platform.types import JobId


class DecentriqDeployment:
    def __init__(
                 self,
                 credentials_file,
                 python_computation_filename,
                 data_clean_room_name,
                 schema1=None,
                 schema2=None
                ):
        """"
        TODO: Is it necessary to abstract for any number of schemas?
        """
        self.credentials_file = credentials_file
        self.data_clean_room_name = data_clean_room_name
        self.python_computation_filename = python_computation_filename
        self.schema1 = self.create_schema(schema1)
        self.schema2 = self.create_schema(schema2)

    def create_schema(self, schema):
        """Convert schema to the format required by Decentriq."""
        if schema is None:
            return None
        mapping = {"int": dqsql.PrimitiveType.INT64,
                   "float": dqsql.PrimitiveType.FLOAT64,
                   "string": dqsql.PrimitiveType.STRING
                   }
        return [(x[0], mapping[x[1]], False) for x in schema]

    def initialize_session(self, credentials_file="credentials"):
        # Get credentials from file
        with open(credentials_file, "r") as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        self.user_email = lines[0]
        api_token = lines[1]

        self.client = dq.create_client(self.user_email, api_token, integrate_with_platform=True)
        self.specs = dq.enclave_specifications.versions([
            "decentriq.driver:v4",
            "decentriq.sql-worker:v4",
            "decentriq.python-ml-worker:v2"
        ])
        self.auth = self.client.platform.create_auth_using_decentriq_pki()
        self.session = self.client.create_session(self.auth, self.specs)

    def publish_data_clean_room(self, second_party_user_email):
        """
        NOTE: With Tabular data, we need to know the columns for all parties at DCR publish time.
        """
        python_builder = dq.DataRoomBuilder(
            self.data_clean_room_name,
            enclave_specs=self.specs
        )

        # Create a data node for each party.
        data_node_builder1 = dqsql.TabularDataNodeBuilder(
            "party_a",
            schema=self.schema1
        )
        data_node_builder1.add_to_builder(
            python_builder,
            authentication=self.client.platform.decentriq_pki_authentication,
            users=[self.user_email]
        )
        data_node_builder2 = dqsql.TabularDataNodeBuilder(
            "party_b",
            schema=self.schema2
        )
        data_node_builder2.add_to_builder(
            python_builder,
            authentication=self.client.platform.decentriq_pki_authentication,
            users=[second_party_user_email]  # NOTE: grant permissions to Party B user
        )

        # Create the python computation node.
        with open(self.python_computation_filename, "rb") as input_script:
            my_script_content_from_file = input_script.read()
        script_node = dq.StaticContent("script_node", my_script_content_from_file)
        script_node_id = python_builder.add_compute_node(script_node)

        training_node = dqc.StaticContainerCompute(
            name="training_node",
            command=["python", "/input/train_script_decentriq.py"],
            mount_points=[
                MountPoint(path="/input/train_script_decentriq.py", dependency=script_node_id),
                MountPoint(path="/input/party_a", dependency="party_a"),
                MountPoint(path="/input/party_b", dependency="party_b")
            ],
            output_path="/output",
            enclave_type="decentriq.python-ml-worker",
            include_container_logs_on_error=True
        )
        self.training_node_id = python_builder.add_compute_node(training_node)

        # Add executtion and retrieval permissions.
        python_builder.add_user_permission(
            email=self.user_email,
            authentication_method=self.client.platform.decentriq_pki_authentication,
            permissions=[
                # NOTE: Permissions for tabular datasets are added from "add_to_builder".
                # dq.Permissions.leaf_crud("party_a"),
                # dq.Permissions.leaf_crud("party_b"),
                dq.Permissions.execute_compute(self.training_node_id),
                dq.Permissions.retrieve_published_datasets(),
                dq.Permissions.update_data_room_status(),
                dq.Permissions.retrieve_data_room_status(),
                dq.Permissions.retrieve_data_room(),
                dq.Permissions.retrieve_audit_log()
            ]
        )

        # Publish Data Clean Room.
        self.data_room = python_builder.build()
        self.python_dcr_id = self.session.publish_data_room(self.data_room)
        print("DCR is successfully published. DCR ID:", self.python_dcr_id)

    def upload_data(self, data_name, data_filename):
        """"Upload the contents of a single tabular dataset."""
        key = dq.Key()

        input_data = dqsql.read_input_csv_file(data_filename, has_header=True, delimiter=",")

        dataset_id = dqsql.upload_and_publish_tabular_dataset(
            input_data,
            key,
            self.python_dcr_id,
            table=data_name,
            session=self.session,
            description=f"These are the {data_name} data",
            validate=True
        )

        # Get dataset from postgres
        self.client.get_dataset(dataset_id)

    def upload_abstract_data(self, data_node_id, local_data_filename, filename_in_dcr):
        """"
        Upload an abstract file to the DCR.

        Args:
            data_node_id: If the DCR has been published through the Web API, this can be obtained through "Expert View".
                          The name seems to follow the pattern "@table/$(DATA_NAME)/dataset".
            local_data_filename: Path of file that is to be uploaded.
            filename_in_dcr: The desired filename for the uploaded file to possess in the DCR.
        Returns:
            None
        """
        key = dq.Key()
        with open(local_data_filename, "r") as file:
            input_data = file.read()
        input_data = io.BytesIO(input_data.encode())
        dataset_id = self.client.upload_dataset(input_data, key, filename_in_dcr)
        self.session.publish_dataset(self.python_dcr_id, dataset_id, data_node_id, key)

    def execute_computations(self, training_node_id, extraction_folder="."):
        """Run computation and get results."""
        job_id = self.session.run_computation(self.python_dcr_id, training_node_id)
        print(f"Job ID for the computation {job_id.id, job_id.compute_node_id}")  # TODO: should we temporary store ids?

        raw_result = self.session.get_computation_result(job_id)
        zip_result = dqc.read_result_as_zipfile(raw_result)
        zip_result.extractall(extraction_folder)

    def get_results_from_job_id(self, job_id, training_node_id, extraction_folder="."):
        """
        Get results when the connection with the DCR is interupted in long-lasting computations.
        """
        job_id = JobId(job_id=job_id, compute_node_id=training_node_id)
        raw_result = self.session.get_computation_result(job_id)
        zip_result = dqc.read_result_as_zipfile(raw_result)
        zip_result.extractall(extraction_folder)
