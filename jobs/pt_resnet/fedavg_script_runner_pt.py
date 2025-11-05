# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import yaml
import argparse  
import sys
from src.mnist_cnn import MnistCnn

from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner


def load_clients_from_yaml(yaml_file_path: str) -> list:
    """Loads a project.yml file and returns a list of client names."""
    try:
        with open(yaml_file_path, 'r') as f:
            project_config = yaml.safe_load(f)

        # Extract participant names that are of type 'client'
        participants = project_config.get('participants', [])
        client_names = [p['name'] for p in participants if p.get('type') == 'client']
        return client_names
        
    except FileNotFoundError:
        print(f"Error: Project YAML file not found at: {yaml_file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing YAML file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    
    # --- Setup Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Create an NVFlare FedAvg job from a project.yml file.")
    parser.add_argument(
        "-p",
        "--project_yaml",
        type=str,
        required=True,
        help="Path to the project.yml file containing participant definitions."
    )
    args = parser.parse_args()

    # --- Load Client Names from the specified file ---
    client_names = load_clients_from_yaml(args.project_yaml)

    # --- Configure Job ---
    n_clients = len(client_names)
    num_rounds = 5 
    train_script = "src/pt_mnist_fl.py"

    if n_clients == 0:
        print("Error: No clients of type 'client' found in project.yml. Exiting.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Configuring job for {n_clients} clients: {client_names}")

    job = FedAvgJob(
        name="pt_mnist_fedavg",
        n_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=MnistCnn(),
    )

    # Add clients by iterating over the names found in the YAML
    for client_name in client_names:
        executor = ScriptRunner(
            script=train_script,
            # You can also use the client_name to dynamically assign script args
            script_args="",  # e.g., f"--batch_size 32 --data_path /tmp/data/{client_name}"
        )
        # Use the specific client_name from the YAML
        job.to(executor, client_name)

    job_output_path = "./job_config"
    job.export_job(job_output_path)
    print(f"\nJob configuration exported to: {job_output_path}")
    
    # job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
