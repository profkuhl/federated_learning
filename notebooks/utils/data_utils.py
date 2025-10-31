import os
import subprocess
import sys
import tempfile
from pathlib import Path
import json  # Added this import

import numpy as np
import torch
from torch.utils.data import Subset

def split_dataset_indices(num_samples: int, num_sites: int, split_method: str):
    """
    Splits a total number of samples among N sites using a specified method.
    
    Returns a list of lists, where each inner list contains the indices
    for that site.
    """
    indices = np.arange(num_samples)
    # Randomize the data order before splitting
    np.random.shuffle(indices) 

    # Calculate partition sizes
    split_sizes = []
    if split_method == "uniform":
        ratio_vec = np.ones(num_sites)
    elif split_method == "linear":
        ratio_vec = np.linspace(1, num_sites, num=num_sites)
    elif split_method == "square":
        ratio_vec = np.square(np.linspace(1, num_sites, num=num_sites))
    elif split_method == "exponential":
        ratio_vec = np.exp(np.linspace(1, num_sites, num=num_sites))
    else:
        raise ValueError(f"Split method '{split_method}' not implemented!")

    total_ratio = sum(ratio_vec)
    left = num_samples
    for i in range(num_sites - 1):
        # Ensure at least 1 sample per site if possible
        x = max(1, int(num_samples * ratio_vec[i] / total_ratio))
        x = min(x, left - (num_sites - 1 - i)) # Ensure future sites get at least 1
        left = left - x
        split_sizes.append(x)
    split_sizes.append(left) # Assign remainder to the last site

    print(f"Splitting {num_samples} samples into {num_sites} sites ({split_method}): {split_sizes}")

    # Split the shuffled indices
    client_indices = []
    current_idx = 0
    for size in split_sizes:
        client_indices.append(indices[current_idx : current_idx + size])
        current_idx += size
        
    return client_indices


def _save_tensors_to_pt(data_tensor, label_tensor, file_path):
    """Helper to save data and label tensors to a .pt file."""
    torch.save((data_tensor, label_tensor), file_path)
    print(f"  Saved {len(label_tensor)} items to {file_path}")


def split_and_distribute(
    train_data,
    train_labels,
    test_data,
    test_labels,
    inventory_path: str,
    split_method: str,
    remote_dest_path: str,
):
    """
    Splits, saves, and distributes any dataset (provided as tensors or arrays) 
    to Ansible clients.
    
    *** This will DELETE and REPLACE the remote_dest_path on all clients. ***

    Args:
        train_data: A PyTorch Tensor or NumPy array of training data (X_train).
        train_labels: A PyTorch Tensor or NumPy array of training labels (y_train).
        test_data: A PyTorch Tensor or NumPy array of testing data (X_test).
        test_labels: A PyTorch Tensor or NumPy array of testing labels (y_test).
        inventory_path: Path to the inventory.ini file.
        split_method: 'uniform', 'exponential', 'square', or 'linear'.
        remote_dest_path: Absolute path on clients (e.g., "/tmp/my_data").
    """
    print("--- Starting Data Split and Distribution ---")

    # --- 1. Validate Inputs & Convert to Tensors ---
    print("Validating inputs...")
    if len(train_data) != len(train_labels):
        print(f"Error: train_data length ({len(train_data)}) != train_labels length ({len(train_labels)})")
        return
    if len(test_data) != len(test_labels):
        print(f"Error: test_data length ({len(test_data)}) != test_labels length ({len(test_labels)})")
        return
    if isinstance(train_data, np.ndarray):
        train_data = torch.from_numpy(train_data)
    if isinstance(train_labels, np.ndarray):
        train_labels = torch.from_numpy(train_labels)
    if isinstance(test_data, np.ndarray):
        test_data = torch.from_numpy(test_data)
    if isinstance(test_labels, np.ndarray):
        test_labels = torch.from_numpy(test_labels)

    # --- 2. Get Client Info from Ansible (THE CORRECT WAY) ---
    if not Path(inventory_path).exists():
        print(f"Error: Inventory file not found at {inventory_path}", file=sys.stderr)
        return

    print(f"Querying Ansible inventory '{inventory_path}' for client list...")
    cmd_inventory = [
        "ansible-inventory", "-i", inventory_path, "--list"
    ]
    try:
        result = subprocess.run(cmd_inventory, check=True, text=True, capture_output=True, timeout=10)
        inventory_data = json.loads(result.stdout)
        
        if "nvflare_clients" not in inventory_data or "hosts" not in inventory_data["nvflare_clients"]:
            print(f"Error: [nvflare_clients] group or its hosts not found in '{inventory_path}' output.", file=sys.stderr)
            return
            
        # This is the robust way to get the hostnames as Ansible sees them
        client_names = inventory_data["nvflare_clients"]["hosts"]
        num_sites = len(client_names)
        if num_sites == 0:
            print(f"Error: No hosts found in [nvflare_clients] group.", file=sys.stderr)
            return
            
        print(f"Found {num_sites} clients: {client_names}")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error querying Ansible inventory:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return
    except json.JSONDecodeError as e:
        print(f"\n❌ Error parsing Ansible inventory output:", file=sys.stderr)
        print(e, file=sys.stderr)
        return

    # --- 3. Create Local Splits in a Temp Directory ---
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nCreating local splits in temporary directory: {temp_dir}")
        
        # 3a. Split Train Dataset
        train_indices = split_dataset_indices(len(train_data), num_sites, split_method)
        
        for i, client_name in enumerate(client_names):
            site_indices = train_indices[i]
            data_slice = train_data[site_indices]
            label_slice = train_labels[site_indices]
            
            # Use the correct client name for the file
            file_name = f"{client_name}_train.pt"
            _save_tensors_to_pt(data_slice, label_slice, Path(temp_dir) / file_name)

        # 3b. Save the FULL Test Dataset
        print(f"\nSaving full test dataset...")
        _save_tensors_to_pt(test_data, test_labels, Path(temp_dir) / "test_data.pt")

        # --- 4. Distribute Files with Ansible (Targeted) ---
        print("\nStarting distribution to clients via Ansible...")
        
        # Command to DELETE the existing directory
        cmd_delete_dir = [
            "ansible", "-i", inventory_path, "nvflare_clients",
            "-m", "file",
            "-a", f"path={remote_dest_path} state=absent"
        ]
        
        # Command to RE-CREATE the directory
        cmd_create_dir = [
            "ansible", "-i", inventory_path, "nvflare_clients",
            "-m", "file",
            "-a", f"path={remote_dest_path} state=directory mode=0755"
        ]

        try:
            print(f"  WARNING: Deleting remote directory '{remote_dest_path}' on all clients...")
            subprocess.run(cmd_delete_dir, check=True, text=True, capture_output=True, timeout=60)
            
            print(f"  Re-creating remote directory '{remote_dest_path}'...")
            subprocess.run(cmd_create_dir, check=True, text=True, capture_output=True, timeout=60)
            
        except Exception as e:
            print(f"\n❌ Error re-creating remote directories:", file=sys.stderr)
            print(e.stderr if hasattr(e, 'stderr') else e, file=sys.stderr)
            return

        # Loop and send files one by one
        print("  Distributing client-specific files...")
        for client_name in client_names:
            print(f"    Sending files to {client_name}...")
            
            # --- Send client's train file ---
            train_file_name = f"{client_name}_train.pt"
            src_train_path = Path(temp_dir) / train_file_name
            dest_train_path = Path(remote_dest_path) / train_file_name
            
            cmd_copy_train = [
                "ansible", "-i", inventory_path, client_name, # Target ONE client
                "-m", "copy",
                "-a", f"src={src_train_path} dest={dest_train_path} mode=0644"
            ]
            
            # --- Send shared test file ---
            src_test_path = Path(temp_dir) / "test_data.pt"
            dest_test_path = Path(remote_dest_path) / "test_data.pt"
            
            cmd_copy_test = [
                "ansible", "-i", inventory_path, client_name, # Target ONE client
                "-m", "copy",
                "-a", f"src={src_test_path} dest={dest_test_path} mode=0644"
            ]

            try:
                # Run both copy commands for this client
                subprocess.run(cmd_copy_train, check=True, text=True, capture_output=True, timeout=60)
                subprocess.run(cmd_copy_test, check=True, text=True, capture_output=True, timeout=60)
                
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Error during Ansible copy to {client_name}:", file=sys.stderr)
                print(e.stderr, file=sys.stderr)
                # Continue to next client
            
        print("\n✅ Successfully distributed all files.")
        
    print("--- Process Complete ---")

