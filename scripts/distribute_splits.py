#!/usr/bin/env python3

import argparse
import configparser
import os
import subprocess
import sys
from pathlib import Path

# --- Configuration ---
# Constants from your data_split.sh
SIZE_TOTAL = 11000000
SIZE_VALID = 1000000
SITE_NAME_PREFIX = "site-"
PREPARE_SCRIPT_PATH = "utils/prepare_data_split.py"
LOCAL_OUTPUT_DIR = "/tmp/nvflare/random_forest/HIGGS/data_splits"

def main():
    parser = argparse.ArgumentParser(
        description="Generate and distribute data split JSON files to Ansible clients.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Absolute path to the main dataset file (e.g., /path/to/HIGGS.csv)"
    )
    parser.add_argument(
        "--split_method",
        type=str,
        required=True,
        choices=["uniform", "exponential", "square", "linear"],
        help="""

        The method to use for splitting the data.

        
        uniform: all clients has the same amount of data
        linear: the amount of data is linearly correlated with the client ID (1 to M)
        square: the amount of data is correlated with the client ID in a squared fashion (1^2 to M^2)
        exponential: the amount of data is correlated with the client ID in an exponential fashion (exp(1) to exp(M))

        The choice of data split depends on dataset and the number of participants.

        For a large dataset like HIGGS, if the number of clients is small (e.g. 5), each client will still have sufficient data to train on with uniform split, and hence exponential would be used to observe the performance drop caused by non-uniform data split. If the number of clients is large (e.g. 20), exponential split will be too aggressive, and linear/square should be used.
        """
    )
    parser.add_argument(
        "--inventory",
        type=str,
        default="inventory.ini",
        help="Path to the Ansible inventory file."
    )
    parser.add_argument(
        "--remote_dest",
        type=str,
        required=True,
        help="Absolute path on the remote clients to copy the split files to."
    )
    args = parser.parse_args()

    # --- 1. Parse Ansible Inventory ---
    print(f"Parsing inventory file: {args.inventory}")
    if not Path(args.inventory).exists():
        print(f"Error: Inventory file not found at {args.inventory}", file=sys.stderr)
        sys.exit(1)
        
    config = configparser.ConfigParser()
    config.read(args.inventory)
    
    if "nvflare_clients" not in config:
        print(f"Error: [nvflare_clients] group not found in {args.inventory}", file=sys.stderr)
        sys.exit(1)
        
    clients = list(config["nvflare_clients"].keys())
    site_num = len(clients)
    print(f"Found {site_num} clients in '[nvflare_clients]' group.")

    # --- 2. Generate Split Files Locally ---
    print("\nStarting local data split generation...")
    
    # Define the output path based on your script's logic
    local_split_dir = Path(LOCAL_OUTPUT_DIR) / f"{site_num}_{args.split_method}"
    
    # Create the local directory
    os.makedirs(local_split_dir, exist_ok=True)
    print(f"Generating split files in: {local_split_dir}")

    # Build the command to call prepare_data_split.py
    cmd_generate = [
        "python3",
        PREPARE_SCRIPT_PATH,
        "--data_path", args.data_path,
        "--site_num", str(site_num),
        "--size_total", str(SIZE_TOTAL),
        "--size_valid", str(SIZE_VALID),
        "--split_method", args.split_method,
        "--out_path", str(local_split_dir),
        "--site_name_prefix", SITE_NAME_PREFIX
    ]

    try:
        subprocess.run(cmd_generate, check=True, text=True)
        print("✅ Successfully generated split files locally.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error generating split files:", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    # --- 3. Distribute Files with Ansible ---
    print("\nStarting distribution to clients via Ansible...")

    # Ensure remote directory exists
    cmd_mkdir = [
        "ansible",
        "-i", args.inventory,
        "nvflare_clients",
        "-m", "file",
        "-a", f"path={args.remote_dest} state=directory mode=0755"
    ]
    
    # Copy the entire directory of split files
    cmd_copy = [
        "ansible",
        "-i", args.inventory,
        "nvflare_clients",
        "-m", "copy",
        "-a", f"src={local_split_dir}/ dest={args.remote_dest} mode=0644"
    ]

    try:
        print("Ensuring remote directory exists...")
        subprocess.run(cmd_mkdir, check=True, text=True, capture_output=True)
        
        print(f"Copying {local_split_dir}/* to {args.remote_dest} on all clients...")
        result = subprocess.run(cmd_copy, check=True, text=True, capture_output=True)
        
        print("\n--- Ansible Output ---")
        print(result.stdout)
        print("----------------------")
        
        print("✅ Successfully distributed split files to all clients.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during Ansible distribution:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
