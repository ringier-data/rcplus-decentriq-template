import decentriq_platform as dq
import decentriq_platform.sql as dqsql
import decentriq_platform.container as dqc
from decentriq_platform.container.proto import MountPoint


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
        TODO: Abstract for any number of schemas?
        TODO: Deprecate hardcoded schemas.
        """
        self.credentials_file = credentials_file
        self.data_clean_room_name = data_clean_room_name
        self.python_computation_filename = python_computation_filename

    def deploy_workflow(self):
        """Handle all stages from the parent for testing purposes. TODO: remove in future"""
        self.initialize_session(self.credentials_file)
        self.publish_data_clean_room()
        self.upload_data(data_name="party_a", data_filename="examples/breast_cancer/data/data_party_a.csv")
        self.upload_data(data_name="party_b", data_filename="examples/breast_cancer/data/data_party_b.csv")
        self.execute_computations()

    def initialize_session(self, credentials_file="credentials"):
        # Get credentials from file
        with open(credentials_file, "r") as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        self.user_email = lines[0]
        api_token = lines[1]

        self.client = dq.create_client(self.user_email, api_token, integrate_with_platform=True)
        self.specs = dq.enclave_specifications.versions([
            "decentriq.driver:v2",
            "decentriq.sql-worker:v2",
            "decentriq.python-ml-worker:v1"
        ])
        self.auth = self.client.platform.create_auth_using_decentriq_pki()
        self.session = self.client.create_session(self.auth, self.specs)

    def publish_data_clean_room(self):
        python_builder = dq.DataRoomBuilder(
            self.data_clean_room_name,
            enclave_specs=self.specs
        )

        # Create a data node for each party.
        data_node_builder1 = dqsql.TabularDataNodeBuilder(
            "party_a",
            schema=[
                ("id", dqsql.PrimitiveType.INT64, False),
                ("mean radius", dqsql.PrimitiveType.FLOAT64, False),
                ("mean texture", dqsql.PrimitiveType.FLOAT64, False),
                ("mean perimeter", dqsql.PrimitiveType.FLOAT64, False)
            ]
        )
        data_node_builder1.add_to_builder(
            python_builder,
            authentication=self.client.platform.decentriq_pki_authentication,
            users=[self.user_email]
        )
        data_node_builder2 = dqsql.TabularDataNodeBuilder(
            "party_b",
            schema=[
                ("id", dqsql.PrimitiveType.INT64, False),
                ("y", dqsql.PrimitiveType.FLOAT64, False)
            ]
        )
        data_node_builder2.add_to_builder(
            python_builder,
            authentication=self.client.platform.decentriq_pki_authentication,
            users=[self.user_email]
        )

        # Create the python computation node.
        with open(self.python_computation_filename, "rb") as input_script:
            my_script_content_from_file = input_script.read()
        script_node1 = dq.StaticContent("python_script", my_script_content_from_file)
        python_builder.add_compute_node(script_node1)

        training_node = dqc.StaticContainerCompute(
            name="training_node",
            command=["python", "/input/train_script_decentriq.py"],
            mount_points=[
                MountPoint(path="/input/train_script_decentriq.py", dependency="python_script"),
                MountPoint(path="/input/party_a", dependency="party_a"),
                MountPoint(path="/input/party_b", dependency="party_b")
            ],
            output_path="/output",
            enclave_type="decentriq.python-ml-worker",
            include_container_logs_on_error=True
        )
        python_builder.add_compute_node(training_node)

        # Add executtion and retrival permissions.
        python_builder.add_user_permission(
            email="alexandros.metsai@ringier.ch",
            authentication_method=self.client.platform.decentriq_pki_authentication,
            permissions=[
                # NOTE: no permissions  needed for tabular datasets?
                # dq.Permissions.leaf_crud("party_a"),
                # dq.Permissions.leaf_crud("party_b"),
                dq.Permissions.execute_compute("training_node"),
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

    def execute_computations(self, extraction_folder="."):
        # Run computation and get results.
        raw_result = self.session.run_computation_and_get_results(self.python_dcr_id, "training_node")
        zip_result = dqc.read_result_as_zipfile(raw_result)
        zip_result.extractall(extraction_folder)
        zip_result.extractall(".")
