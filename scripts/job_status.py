#!/usr/bin/env python3
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
NVFlare Job Status Monitor

Displays graphical representation of job progress, status, and accuracy metrics.

Usage:
    python job_status.py <job_id>
    python job_status.py <job_id> --watch  # Continuous monitoring
"""

import argparse
import subprocess
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"


def get_job_status(job_id: str, admin_dir: str) -> dict:
    """Get job status from NVFlare API."""
    script = f'''
from nvflare.fuel.flare_api.flare_api import new_secure_session
import json

sess = new_secure_session(
    username="admin@nvidia.com",
    startup_kit_location="{admin_dir}"
)

try:
    status = sess.get_job_meta("{job_id}")
    print(json.dumps(status))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
finally:
    sess.close()
'''
    try:
        result = subprocess.run(
            ["/opt/nvflare_provision_venv/bin/python3", "-c", script],
            capture_output=True,
            text=True,
            timeout=30
        )
        import json
        # Find the JSON in output (skip "Connecting to FLARE ..." messages)
        for line in result.stdout.strip().split('\n'):
            if line.startswith('{'):
                return json.loads(line)
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Could not get job status"}


def get_server_logs(job_id: str) -> str:
    """Get NVFlare server logs filtered by job_id."""
    try:
        result = subprocess.run(
            ["docker", "logs", "nvflare-server", "--tail", "5000"],
            capture_output=True,
            text=True,
            timeout=60
        )
        # Combine stdout and stderr
        all_logs = result.stdout + result.stderr
        # Filter lines containing the job_id
        lines = [line for line in all_logs.split('\n') if job_id in line]
        return '\n'.join(lines)
    except Exception as e:
        return ""


def get_round_info_from_logs(job_id: str) -> list:
    """Get round start information from logs."""
    try:
        result = subprocess.run(
            ["docker", "logs", "nvflare-server"],
            capture_output=True,
            text=True,
            timeout=60
        )
        all_logs = result.stdout + result.stderr

        # Find all "Round X started" lines
        rounds = []
        in_job = False
        for line in all_logs.split('\n'):
            if job_id in line:
                in_job = True
            if in_job and "Round" in line and "started" in line:
                match = re.search(r"Round (\d+) started", line)
                if match:
                    round_num = int(match.group(1))
                    if round_num not in rounds:
                        rounds.append(round_num)
            # Reset if we see a different job
            if "run=" in line and job_id not in line and in_job:
                # Check if this is a new job start
                if "Workflow" in line and "started" in line:
                    in_job = False
        return sorted(rounds)
    except Exception as e:
        return []


def parse_logs(logs: str, job_id: str) -> dict:
    """Parse server logs to extract job metrics."""
    data = {
        "rounds_started": [],
        "client_results": 0,
        "validation_metrics": defaultdict(list),
        "best_metric": None,
        "best_round": None,
        "clients": set(),
    }

    # Find rounds started
    round_pattern = r"Round (\d+) started"
    for match in re.finditer(round_pattern, logs):
        round_num = int(match.group(1))
        if round_num not in data["rounds_started"]:
            data["rounds_started"].append(round_num)

    # Count client results
    result_pattern = r"got result from client (\S+)"
    for match in re.finditer(result_pattern, logs):
        data["client_results"] += 1
        data["clients"].add(match.group(1))

    # Extract validation metrics
    metric_pattern = r"validation metric ([\d.]+) from client (\S+)"
    for match in re.finditer(metric_pattern, logs):
        metric = float(match.group(1))
        client = match.group(2)
        data["validation_metrics"][client].append(metric)

    # Find best metric
    best_pattern = r"new best validation metric at round (\d+): ([\d.]+)"
    for match in re.finditer(best_pattern, logs):
        round_num = int(match.group(1))
        metric = float(match.group(2))
        if data["best_metric"] is None or metric > data["best_metric"]:
            data["best_metric"] = metric
            data["best_round"] = round_num

    return data


def progress_bar(current: int, total: int, width: int = 30, fill: str = "█", empty: str = "░") -> str:
    """Create a progress bar string."""
    if total == 0:
        return f"[{empty * width}]"

    filled = int(width * current / total)
    bar = fill * filled + empty * (width - filled)
    percent = 100 * current / total
    return f"[{bar}] {percent:5.1f}%"


def accuracy_bar(accuracy: float, width: int = 20) -> str:
    """Create an accuracy bar (0-100% scale)."""
    filled = int(width * accuracy)
    bar = "▓" * filled + "░" * (width - filled)
    return f"[{bar}]"


def print_header(job_id: str, status: str):
    """Print the header section."""
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  NVFlare Job Status Monitor{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 70}{Colors.RESET}")
    print()
    print(f"  {Colors.BOLD}Job ID:{Colors.RESET}  {job_id}")

    # Color-coded status
    if "COMPLETED" in status:
        status_color = Colors.GREEN
        status_icon = "✓"
    elif "RUNNING" in status:
        status_color = Colors.YELLOW
        status_icon = "●"
    elif "FAILED" in status or "EXCEPTION" in status:
        status_color = Colors.RED
        status_icon = "✗"
    else:
        status_color = Colors.GRAY
        status_icon = "○"

    print(f"  {Colors.BOLD}Status:{Colors.RESET}  {status_color}{status_icon} {status}{Colors.RESET}")
    print()


def print_progress(data: dict, num_rounds: int = 10, num_clients: int = 6):
    """Print progress section."""
    current_round = max(data["rounds_started"]) + 1 if data["rounds_started"] else 0
    total_expected = num_rounds * num_clients

    print(f"  {Colors.BOLD}Progress{Colors.RESET}")
    print(f"  {'─' * 50}")

    # Round progress
    print(f"  Rounds:    {progress_bar(current_round, num_rounds)} {current_round}/{num_rounds}")

    # Client results progress
    print(f"  Results:   {progress_bar(data['client_results'], total_expected)} {data['client_results']}/{total_expected}")

    # Clients participating
    print(f"  Clients:   {len(data['clients'])}/{num_clients} participating")
    print()


def print_accuracy(data: dict):
    """Print accuracy section."""
    print(f"  {Colors.BOLD}Accuracy Metrics{Colors.RESET}")
    print(f"  {'─' * 50}")

    if data["best_metric"]:
        best_pct = data["best_metric"] * 100
        print(f"  {Colors.GREEN}★ Best:{Colors.RESET}     {accuracy_bar(data['best_metric'])} {best_pct:5.2f}% (Round {data['best_round']})")

    print()

    if data["validation_metrics"]:
        print(f"  {Colors.BOLD}Latest Client Accuracies:{Colors.RESET}")

        # Sort clients by latest accuracy (descending)
        sorted_clients = sorted(
            data["validation_metrics"].keys(),
            key=lambda c: data["validation_metrics"][c][-1] if data["validation_metrics"][c] else 0,
            reverse=True
        )

        for client in sorted_clients:
            metrics = data["validation_metrics"][client]
            if metrics:
                latest = metrics[-1]
                pct = latest * 100
                # Truncate or pad client name for display
                if len(client) > 20:
                    display_name = client[:17] + "..."
                else:
                    display_name = client
                display_name = display_name.ljust(20)
                print(f"    {display_name} {accuracy_bar(latest)} {pct:5.2f}%")
    print()


def print_footer():
    """Print footer."""
    print(f"{Colors.GRAY}  {'─' * 50}{Colors.RESET}")
    print(f"{Colors.GRAY}  Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
    print()


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor NVFlare job status with graphical representation"
    )
    parser.add_argument("job_id", help="The job ID to monitor")
    parser.add_argument(
        "-w", "--watch",
        action="store_true",
        help="Continuously monitor the job (updates every 10 seconds)"
    )
    parser.add_argument(
        "-i", "--interval",
        type=int,
        default=10,
        help="Update interval in seconds for watch mode (default: 10)"
    )
    parser.add_argument(
        "-r", "--rounds",
        type=int,
        default=10,
        help="Total number of rounds expected (default: 10)"
    )
    parser.add_argument(
        "-c", "--clients",
        type=int,
        default=6,
        help="Total number of clients expected (default: 6)"
    )
    parser.add_argument(
        "--admin-dir",
        type=str,
        default="workspace/example_project/prod_00/admin@nvidia.com",
        help="Path to admin startup kit"
    )

    args = parser.parse_args()

    # Find admin dir relative to script location or cwd
    admin_dir = args.admin_dir
    if not Path(admin_dir).exists():
        # Try relative to federated_learning directory
        script_dir = Path(__file__).parent.parent
        admin_dir = script_dir / args.admin_dir
        if not admin_dir.exists():
            print(f"{Colors.RED}Error: Admin directory not found: {args.admin_dir}{Colors.RESET}")
            sys.exit(1)
        admin_dir = str(admin_dir)

    try:
        while True:
            if args.watch:
                clear_screen()

            # Get job status
            status_data = get_job_status(args.job_id, admin_dir)
            status = status_data.get("status", "UNKNOWN")

            if "error" in status_data:
                print(f"{Colors.RED}Error: {status_data['error']}{Colors.RESET}")
                if not args.watch:
                    sys.exit(1)
                time.sleep(args.interval)
                continue

            # Get and parse logs
            logs = get_server_logs(args.job_id)
            data = parse_logs(logs, args.job_id)

            # Also get round info separately for more accuracy
            round_info = get_round_info_from_logs(args.job_id)
            if round_info:
                data["rounds_started"] = round_info

            # Display
            print_header(args.job_id, status)
            print_progress(data, args.rounds, args.clients)
            print_accuracy(data)
            print_footer()

            if not args.watch:
                break

            # Check if job is finished
            if "FINISHED" in status:
                print(f"{Colors.GREEN}  Job completed. Exiting watch mode.{Colors.RESET}")
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}  Monitoring stopped.{Colors.RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
