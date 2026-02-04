#!/usr/bin/env python3
"""
Distribute dataset splits to NVFlare clients.

Supports multiple dataset types:
- mnist: Downloads and splits MNIST dataset
- cifar10: Downloads and splits CIFAR-10 dataset
- custom: Uses pre-existing .pt files from a directory

Usage examples:
    # Distribute MNIST with uniform split
    python distribute_splits.py --dataset mnist --split_method uniform

    # Distribute MNIST with non-IID (square) split
    python distribute_splits.py --dataset mnist --split_method square

    # Distribute custom dataset
    python distribute_splits.py --dataset custom --data_path /path/to/data.pt --split_method uniform
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm


def get_clients_from_project_yml(project_yml_path: str) -> list:
    """Read client names from project.yml file."""
    with open(project_yml_path, 'r') as f:
        project = yaml.safe_load(f)

    clients = []
    for participant in project.get('participants', []):
        if participant.get('type') == 'client':
            clients.append(participant['name'])
    return clients


def get_remote_clients_from_inventory(inventory_path: str) -> list:
    """Get list of remote clients from Ansible inventory."""
    cmd = ["ansible-inventory", "-i", inventory_path, "--list"]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=10)
    inventory_data = json.loads(result.stdout)
    return inventory_data.get("nvflare_clients", {}).get("hosts", [])


def is_local_client(client_name: str, remote_clients: list) -> bool:
    """Check if client is local (not in remote clients list)."""
    return client_name not in remote_clients


def split_indices(num_samples: int, num_sites: int, split_method: str) -> list:
    """
    Split indices among sites using specified method.

    Methods:
        uniform: Equal splits
        linear: Linear distribution (1, 2, 3, ... N)
        square: Quadratic distribution (1², 2², 3², ... N²)
        exponential: Exponential distribution (e¹, e², ... eⁿ)
    """
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    if split_method == "uniform":
        ratio_vec = np.ones(num_sites)
    elif split_method == "linear":
        ratio_vec = np.linspace(1, num_sites, num=num_sites)
    elif split_method == "square":
        ratio_vec = np.square(np.linspace(1, num_sites, num=num_sites))
    elif split_method == "exponential":
        ratio_vec = np.exp(np.linspace(1, num_sites, num=num_sites))
    else:
        raise ValueError(f"Unknown split method: {split_method}")

    total_ratio = sum(ratio_vec)
    split_sizes = []
    left = num_samples

    for i in range(num_sites - 1):
        size = max(1, int(num_samples * ratio_vec[i] / total_ratio))
        size = min(size, left - (num_sites - 1 - i))
        left -= size
        split_sizes.append(size)
    split_sizes.append(left)

    client_indices = []
    current = 0
    for size in split_sizes:
        client_indices.append(indices[current:current + size])
        current += size

    return client_indices


def load_mnist():
    """Download and load MNIST dataset."""
    from torchvision import datasets, transforms

    print("\n📥 Downloading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_ds = datasets.MNIST(root='/tmp/mnist_raw', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root='/tmp/mnist_raw', train=False, download=True, transform=transform)

    print("🔄 Converting training data to tensors...")
    train_data = torch.stack([train_ds[i][0] for i in tqdm(range(len(train_ds)), desc="   Train", unit="img")])
    train_labels = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])

    print("🔄 Converting test data to tensors...")
    test_data = torch.stack([test_ds[i][0] for i in tqdm(range(len(test_ds)), desc="   Test", unit="img")])
    test_labels = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])

    return train_data, train_labels, test_data, test_labels


def load_cifar10():
    """Download and load CIFAR-10 dataset."""
    from torchvision import datasets, transforms

    print("\n📥 Downloading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_ds = datasets.CIFAR10(root='/tmp/cifar10_raw', train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root='/tmp/cifar10_raw', train=False, download=True, transform=transform)

    print("🔄 Converting training data to tensors...")
    train_data = torch.stack([train_ds[i][0] for i in tqdm(range(len(train_ds)), desc="   Train", unit="img")])
    train_labels = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])

    print("🔄 Converting test data to tensors...")
    test_data = torch.stack([test_ds[i][0] for i in tqdm(range(len(test_ds)), desc="   Test", unit="img")])
    test_labels = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])

    return train_data, train_labels, test_data, test_labels


def load_custom(data_path: str):
    """Load custom dataset from .pt file containing (train_data, train_labels, test_data, test_labels)."""
    print(f"\n📥 Loading custom dataset from {data_path}...")
    data = torch.load(data_path)
    if len(data) == 4:
        print("   ✓ Dataset loaded successfully")
        return data
    else:
        raise ValueError("Custom dataset must contain (train_data, train_labels, test_data, test_labels)")


def distribute_data(
    clients: list,
    remote_clients: list,
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
    split_method: str,
    inventory_path: str,
    local_data_dir: str,
    remote_data_subdir: str,
):
    """Split and distribute data to all clients."""
    import shutil

    print("\n" + "=" * 60)
    print("📊 SPLITTING DATA")
    print("=" * 60)

    # Split training data
    client_indices = split_indices(len(train_data), len(clients), split_method)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save client training splits with progress bar
        print("\n💾 Creating client data splits...")
        for i, client in enumerate(tqdm(clients, desc="   Splitting", unit="client")):
            indices = client_indices[i]
            data_slice = train_data[indices]
            label_slice = train_labels[indices]

            file_path = temp_path / f"{client}_train.pt"
            torch.save((data_slice, label_slice), file_path)

        # Save shared test data
        test_file = temp_path / "test_data.pt"
        torch.save((test_data, test_labels), test_file)
        print(f"   ✓ Created test_data.pt ({len(test_labels)} samples)")

        # Print split summary
        print("\n📋 Split Summary:")
        print("   " + "-" * 40)
        for i, client in enumerate(clients):
            samples = len(client_indices[i])
            bar_len = int(samples / len(train_data) * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"   {client:25s} [{bar}] {samples:,} samples")
        print("   " + "-" * 40)

        # Distribute to local clients
        local_clients = [c for c in clients if is_local_client(c, remote_clients)]
        if local_clients:
            print("\n" + "=" * 60)
            print("🏠 DISTRIBUTING TO LOCAL CLIENTS")
            print("=" * 60)
            local_dest = Path(local_data_dir) / remote_data_subdir
            local_dest.mkdir(parents=True, exist_ok=True)

            # Copy test data
            shutil.copy2(temp_path / "test_data.pt", local_dest / "test_data.pt")

            # Copy each local client's training data
            for client in tqdm(local_clients, desc="   Local copy", unit="client"):
                src = temp_path / f"{client}_train.pt"
                shutil.copy2(src, local_dest / f"{client}_train.pt")

            print(f"   ✓ Copied to {local_dest}")

        # Distribute to remote clients via Ansible
        remote_in_project = [c for c in clients if c in remote_clients]
        if remote_in_project:
            print("\n" + "=" * 60)
            print("🌐 DISTRIBUTING TO REMOTE CLIENTS")
            print("=" * 60)

            # Use tqdm for remote client distribution
            pbar = tqdm(remote_in_project, desc="   Copying", unit="client")
            for client in pbar:
                pbar.set_postfix_str(f"{client}")

                # Get client's home directory from inventory
                remote_dest = f"/home/{client}/nvflare_data/{remote_data_subdir}"

                # Create directory
                subprocess.run([
                    "ansible", "-i", inventory_path, client,
                    "-m", "file",
                    "-a", f"path={remote_dest} state=directory mode=0755"
                ], check=True, capture_output=True)

                # Copy training data
                subprocess.run([
                    "ansible", "-i", inventory_path, client,
                    "-m", "copy",
                    "-a", f"src={temp_path}/{client}_train.pt dest={remote_dest}/{client}_train.pt mode=0644"
                ], check=True, capture_output=True)

                # Copy test data
                subprocess.run([
                    "ansible", "-i", inventory_path, client,
                    "-m", "copy",
                    "-a", f"src={temp_path}/test_data.pt dest={remote_dest}/test_data.pt mode=0644"
                ], check=True, capture_output=True)

            print(f"   ✓ All remote clients received data")

    print("\n" + "=" * 60)
    print("✅ DATA DISTRIBUTION COMPLETE")
    print("=" * 60)
    print(f"   Dataset: {remote_data_subdir}")
    print(f"   Split method: {split_method}")
    print(f"   Total clients: {len(clients)}")
    print(f"   Local clients: {len(local_clients)}")
    print(f"   Remote clients: {len(remote_in_project)}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Distribute dataset splits to NVFlare clients.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["mnist", "cifar10", "custom"],
        help="Dataset to distribute"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to custom dataset .pt file (required if --dataset=custom)"
    )
    parser.add_argument(
        "--split_method",
        type=str,
        default="uniform",
        choices=["uniform", "linear", "square", "exponential"],
        help="Data split method (uniform=IID, others=non-IID)"
    )
    parser.add_argument(
        "--inventory",
        type=str,
        default="ansible/inventory.ini",
        help="Path to Ansible inventory file"
    )
    parser.add_argument(
        "--project_yml",
        type=str,
        default="project.yml",
        help="Path to NVFlare project.yml file"
    )
    parser.add_argument(
        "--local_data_dir",
        type=str,
        default=None,
        help="Local data directory (default: ~/nvflare_data)"
    )

    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.parent  # federated_learning/
    inventory_path = script_dir / args.inventory if not Path(args.inventory).is_absolute() else Path(args.inventory)
    project_yml_path = script_dir / args.project_yml if not Path(args.project_yml).is_absolute() else Path(args.project_yml)

    if not inventory_path.exists():
        print(f"Error: Inventory file not found: {inventory_path}", file=sys.stderr)
        sys.exit(1)

    if not project_yml_path.exists():
        print(f"Error: project.yml not found: {project_yml_path}", file=sys.stderr)
        sys.exit(1)

    # Get clients
    clients = get_clients_from_project_yml(str(project_yml_path))
    remote_clients = get_remote_clients_from_inventory(str(inventory_path))

    # Set local data directory
    local_data_dir = args.local_data_dir or str(Path.home() / "nvflare_data")

    # Print header
    print("\n" + "=" * 60)
    print("🚀 NVFLARE DATA DISTRIBUTION")
    print("=" * 60)
    print(f"   Dataset:      {args.dataset}")
    print(f"   Split method: {args.split_method}")
    print(f"   Inventory:    {inventory_path}")
    print(f"   Project:      {project_yml_path}")
    print("=" * 60)

    print(f"\n👥 Clients from project.yml ({len(clients)}):")
    local_client_list = [c for c in clients if c not in remote_clients]
    for c in clients:
        location = "🏠 local" if c in local_client_list else "🌐 remote"
        print(f"   • {c} ({location})")

    # Load dataset
    if args.dataset == "mnist":
        train_data, train_labels, test_data, test_labels = load_mnist()
        data_subdir = "mnist"
    elif args.dataset == "cifar10":
        train_data, train_labels, test_data, test_labels = load_cifar10()
        data_subdir = "cifar10"
    elif args.dataset == "custom":
        if not args.data_path:
            print("Error: --data_path required for custom dataset", file=sys.stderr)
            sys.exit(1)
        train_data, train_labels, test_data, test_labels = load_custom(args.data_path)
        data_subdir = Path(args.data_path).stem

    print(f"\n📊 Dataset Statistics:")
    print(f"   Training samples: {len(train_data):,}")
    print(f"   Test samples:     {len(test_data):,}")
    print(f"   Data shape:       {list(train_data.shape[1:])}")

    # Distribute
    distribute_data(
        clients=clients,
        remote_clients=remote_clients,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        split_method=args.split_method,
        inventory_path=str(inventory_path),
        local_data_dir=local_data_dir,
        remote_data_subdir=data_subdir,
    )


if __name__ == "__main__":
    main()
