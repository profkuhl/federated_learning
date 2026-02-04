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

"""
Generate NVFlare FedAvg job for CIFAR-10 with ResNet-18.
GPU-intensive job designed for ~10 minute runtime on a 6-client cluster.
"""

import yaml
import argparse
import sys

import torch
from torch import nn
from torchvision.models import resnet18

from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner


def create_cifar10_resnet18():
    """Create ResNet-18 model adapted for CIFAR-10 (32x32 images)."""
    model = resnet18(weights=None, num_classes=10)
    # Adjust first conv layer for smaller CIFAR-10 images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for small images
    return model


def load_clients_from_yaml(yaml_file_path: str) -> list:
    """Loads a project.yml file and returns a list of client names."""
    try:
        with open(yaml_file_path, 'r') as f:
            project_config = yaml.safe_load(f)

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

    parser = argparse.ArgumentParser(
        description="Create NVFlare FedAvg job for CIFAR-10 with ResNet-18."
    )
    parser.add_argument(
        "-p", "--project_yaml",
        type=str,
        required=True,
        help="Path to the project.yml file containing participant definitions."
    )
    parser.add_argument(
        "-r", "--num_rounds",
        type=int,
        default=10,
        help="Number of federated learning rounds (default: 10)"
    )
    args = parser.parse_args()

    # Load client names from project.yml
    client_names = load_clients_from_yaml(args.project_yaml)

    n_clients = len(client_names)
    num_rounds = args.num_rounds
    train_script = "src/pt_cifar10_fl.py"

    if n_clients == 0:
        print("Error: No clients found in project.yml. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"CIFAR-10 + ResNet-18 FedAvg Job Configuration")
    print(f"{'='*60}")
    print(f"Clients: {n_clients}")
    for name in client_names:
        print(f"  - {name}")
    print(f"FL Rounds: {num_rounds}")
    print(f"Training Script: {train_script}")
    print(f"Model: ResNet-18 (~11M parameters)")
    print(f"Expected runtime: ~{num_rounds * 1.5:.0f} minutes")
    print(f"{'='*60}\n")

    # Create initial model
    initial_model = create_cifar10_resnet18()

    job = FedAvgJob(
        name="pt_cifar10_resnet18_fedavg",
        n_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=initial_model,
    )

    # Add clients
    for client_name in client_names:
        executor = ScriptRunner(
            script=train_script,
            script_args="",
        )
        job.to(executor, client_name)

    job_output_path = "./job_config_cifar10"
    job.export_job(job_output_path)
    print(f"Job configuration exported to: {job_output_path}")
