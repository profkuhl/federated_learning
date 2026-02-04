# NVFlare Distributed Deployment Runbook

This guide walks through deploying a dockerized NVFlare federated learning cluster across multiple physical machines, from initial setup to running your first training job.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Deployment Steps](#deployment-steps)
- [Verification](#verification)
- [Running Training Jobs](#running-training-jobs)
- [Troubleshooting](#troubleshooting)
- [Updating Versions](#updating-versions)

---

## Prerequisites

### 1. Hardware Requirements

Each node in the cluster requires:
- NVIDIA GPU (tested with RTX 3090, A100, etc.)
- NVIDIA Driver 570.x or higher
- 16GB+ RAM recommended
- 50GB+ free disk space for Docker images

### 2. Verify Cluster Connectivity

Test SSH access from the server to all client nodes:

```bash
ansible -i ansible/inventory.ini all -m ping
```

**Expected**: All nodes should return `pong`.

### 3. Verify Docker Installation

Check Docker is installed and running on all nodes:

```bash
ansible -i ansible/inventory.ini all -a "docker --version"
ansible -i ansible/inventory.ini all -a "systemctl is-active docker"
```

**Expected**: Docker version output and `active` status from all nodes.

### 4. Verify NVIDIA GPU Access

Check GPUs are available on all nodes:

```bash
ansible -i ansible/inventory.ini all -a "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"
```

**Expected**: GPU name and driver version from all nodes.

If NVIDIA drivers are not installed:
```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/install_nvidia_drivers.yml
```

### 5. Verify Docker GPU Access

This is the critical test - Docker must be able to access GPUs:

```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/validate_docker_gpu.yml
```

This playbook checks:
- Docker is installed and running
- nvidia-smi is available
- NVIDIA Container Toolkit is installed
- Docker can run GPU workloads

**If Docker cannot access GPUs**, install NVIDIA Container Toolkit:

```bash
ansible -i ansible/inventory.ini all -b -m shell -a "
  distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID) && \
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
  curl -s -L https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
  apt-get update && \
  apt-get install -y nvidia-container-toolkit && \
  nvidia-ctk runtime configure --runtime=docker && \
  systemctl restart docker
"
```

Then re-run the validation playbook.

---

## Configuration

### 1. Configure Participants (`project.yml`)

Edit `project.yml` to define your server and clients:

```yaml
api_version: 3
name: example_project
description: My federated learning project

participants:
  # Server (always required)
  - name: server
    type: server
    org: nvidia
    fed_learn_port: 8002
    admin_port: 8003

  # Local client on server node (optional but recommended for testing)
  - name: k3s-server-07-client
    type: client
    org: nvidia

  # Remote clients
  - name: k3s-client-08
    type: client
    org: nvidia
  - name: k3s-client-09
    type: client
    org: nvidia
  # Add more clients as needed...

  # Admin user for job submission
  - name: admin@nvidia.com
    type: admin
    org: nvidia
    role: project_admin

builders:
  - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
    args:
      template_file:
        - master_template.yml
  - path: nvflare.lighter.impl.static_file.StaticFileBuilder
    args:
      config_folder: config
      docker_image: nvflare-pt-docker  # Must match image built by Ansible
      overseer_agent:
        path: nvflare.ha.dummy_overseer_agent.DummyOverseerAgent
        overseer_exists: false
        args:
          sp_end_point: server:8002:8003
  - path: nvflare.lighter.impl.cert.CertBuilder
  - path: nvflare.lighter.impl.signature.SignatureBuilder
```

### 2. Configure Node IPs (`ansible/inventory.ini`)

Edit `ansible/inventory.ini` with your node IPs and SSH users:

```ini
[nvflare_server]
172.19.1.7  ansible_user=k3s-server-07 ansible_connection=local

[nvflare_clients]
k3s-client-08 ansible_host=172.19.1.8  ansible_user=k3s-client-08
k3s-client-09 ansible_host=172.19.1.9  ansible_user=k3s-client-09
# Add more clients...

[all:vars]
nvflare_container_name=nvflare-pt-docker
nvflare_server_port=8002
nvflare_admin_port=8003
nvflare_server_ip=172.19.1.7
nvflare_workspace_dir=/home/{{ ansible_user }}/nvflare_workspace
nvflare_data_dir=/home/{{ ansible_user }}/nvflare_data
```

**Important**:
- Client names in `inventory.ini` must match client names in `project.yml`
- The server runs with `ansible_connection=local` (no SSH needed)

### 3. Verify Docker Image Configuration (`ansible/Dockerfile`)

The Dockerfile defines the container environment:

```dockerfile
# Use NVIDIA PyTorch container compatible with driver 570.x
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3
FROM ${BASE_IMAGE}

# Install NVFlare and dependencies
RUN python3 -m pip install -U pip setuptools wheel && \
    python3 -m pip install nvflare==2.7.1 && \
    python3 -m pip install torch torchvision numpy pandas polars

WORKDIR /workspace/
```

**Note**: The base image `pytorch:24.12-py3` is compatible with NVIDIA driver 570.x. If you have driver 580.95+, you can use newer images like `pytorch:25.10-py3`.

---

## Deployment Steps

### Option A: Full Automated Deployment (Recommended)

Run the master playbook that orchestrates all deployment steps:

```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/setup_nvflare_networking.yml
```

This single command executes these steps in order:
1. **Configure UFW firewall** - Opens ports 8002/8003, allows internal subnet
2. **Configure /etc/hosts** - Adds hostname resolution for all participants
3. **Deploy NVFlare containers** - Provisions workspace, builds images, distributes startup kits, starts containers
4. **Validate deployment** - Checks containers, ports, and connectivity

**Expected Duration**: 15-30 minutes (mostly Docker image build time)

### Option B: Step-by-Step Deployment

For more control or debugging, run each step individually:

#### Step 1: Configure Firewall

```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/configure_ufw_nvflare.yml
```

Opens ports:
- 8002/tcp - Federated learning communication
- 8003/tcp - Admin console
- 172.19.1.0/24 - Internal subnet communication

#### Step 2: Configure Hostname Resolution

```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/configure_hosts_file.yml
```

Adds entries to `/etc/hosts` on all nodes for:
- `server` → server IP
- `mylocalhost` → server IP (required by NVFlare)
- All client hostnames → their IPs

#### Step 3: Deploy NVFlare Containers

```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/deploy_docker_nvflare.yml
```

This is the main deployment playbook. It:

**On the server node:**
1. Creates NVFlare provisioning virtualenv (`/opt/nvflare_provision_venv`)
2. Runs `nvflare provision` to generate workspace with certificates
3. Builds Docker image `nvflare-pt-docker:latest`
4. Copies server startup kit to `~/nvflare_workspace/server/`
5. Starts `nvflare-server` container with host networking
6. Copies local client startup kit (if configured)
7. Starts `nvflare-client-local` container

**On each client node:**
1. Copies Dockerfile and builds `nvflare-pt-docker:latest` image
2. Receives startup kit via rsync from server
3. Starts `nvflare-client-XX` container with host networking

#### Step 4: Validate Deployment

```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/validate_nvflare_network.yml
```

Checks:
- All containers are running
- Containers use host networking
- Restart policy is `unless-stopped`
- Ports 8002/8003 are listening
- Network connectivity between nodes

---

## Verification

### 1. Check All Containers Are Running

**Server:**
```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

**Expected output:**
```
NAMES                STATUS          PORTS
nvflare-server       Up X hours
nvflare-client-local Up X hours
```

**All nodes:**
```bash
ansible -i ansible/inventory.ini all -a "docker ps --format 'table {{.Names}}\t{{.Status}}'"
```

### 2. Check Container Logs

**Server logs:**
```bash
docker logs nvflare-server --tail 50
```

**Look for:**
- `FL server  starting...`
- `Server started`
- Client connection messages

**Client logs:**
```bash
docker logs nvflare-client-09 --tail 50
```

**Look for:**
- `Starting client...`
- `Client connected to server`

### 3. Verify GPU Access Inside Containers

**Server:**
```bash
docker exec nvflare-server nvidia-smi
```

**Clients:**
```bash
ansible -i ansible/inventory.ini nvflare_clients -m shell -a "
  container_name=\$(docker ps --format '{{.Names}}' | head -1)
  docker exec \$container_name nvidia-smi --query-gpu=name --format=csv,noheader
"
```

### 4. Verify Restart Policy

Containers should persist through reboots:

```bash
ansible -i ansible/inventory.ini all -m shell -a "
  docker inspect --format='{{.Name}}: {{.HostConfig.RestartPolicy.Name}}' \$(docker ps -q)
"
```

**Expected**: All containers show `unless-stopped`

---

## Running Training Jobs

### Step 1: Distribute Training Data

Before running any job, distribute the dataset to all clients:

**MNIST (simple, quick):**
```bash
python scripts/distribute_splits.py --dataset mnist --split_method uniform
```

**CIFAR-10 (larger, for GPU testing):**
```bash
python scripts/distribute_splits.py --dataset cifar10 --split_method square
```

**Split methods:**
| Method | Description | Use Case |
|--------|-------------|----------|
| `uniform` | Equal data per client (IID) | Baseline, fair comparison |
| `linear` | Linearly increasing amounts | Mild heterogeneity |
| `square` | Quadratically increasing | Moderate heterogeneity |
| `exponential` | Exponentially increasing | Extreme heterogeneity |

The script:
1. Downloads the dataset (if needed)
2. Splits it according to the method
3. Copies each client's portion to `~/nvflare_data/{dataset}/` on that node
4. Copies shared test data to all nodes

**Verify data was distributed:**
```bash
ansible -i ansible/inventory.ini all -a "ls -lh ~/nvflare_data/mnist/"
```

### Step 2: Submit a Job

**Option A: Submit pre-configured job (recommended):**

```bash
# Simple MNIST job (5 rounds, ~2-3 minutes)
nvflare job submit -j jobs/configs/pt_mnist_fedavg

# GPU-intensive CIFAR-10 job (10 rounds, ~10-15 minutes)
nvflare job submit -j jobs/configs/pt_cifar10_resnet18
```

**Option B: Generate and submit custom job:**

```bash
# Generate job config from project.yml
cd jobs/generators
python fedavg_mnist.py -p ../../project.yml

# Submit the generated job
nvflare job submit -j ../job_config/pt_mnist_fedavg
```

**Option C: Submit via Python API:**

```python
from nvflare.fuel.flare_api.flare_api import new_secure_session

sess = new_secure_session(
    username="admin@nvidia.com",
    startup_kit_location="workspace/example_project/prod_00/admin@nvidia.com"
)

job_id = sess.submit_job("jobs/configs/pt_mnist_fedavg")
print(f"Submitted job: {job_id}")

sess.close()
```

### Step 3: Monitor Job Progress

**Real-time monitoring with progress bars:**

```bash
# One-time status check
python scripts/job_status.py <job_id>

# Continuous monitoring (updates every 10 seconds)
python scripts/job_status.py <job_id> --watch

# Custom settings
python scripts/job_status.py <job_id> --watch --rounds 10 --clients 6 --interval 5
```

**Example output:**
```
══════════════════════════════════════════════════════════════════════
  NVFlare Job Status Monitor
══════════════════════════════════════════════════════════════════════

  Job ID:  e51d5cb2-2d55-48f8-8f9b-133dfc9a57c3
  Status:  ● RUNNING

  Progress
  ──────────────────────────────────────────────────
  Rounds:    [████████████░░░░░░░░░░░░░░░░░░]  40.0% 4/10
  Results:   [████████████░░░░░░░░░░░░░░░░░░]  40.0% 24/60
  Clients:   6/6 participating

  Accuracy Metrics
  ──────────────────────────────────────────────────
  ★ Best:     [▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░] 56.08% (Round 4)
```

**Check logs directly:**
```bash
# Server logs
docker logs nvflare-server --tail 100

# Client logs
docker logs nvflare-client-09 --tail 100
```

### Step 4: Retrieve Results

After job completion, results are stored in the server's workspace:

```bash
# List completed jobs
ls ~/nvflare_workspace/server/transfer/

# View job results
ls ~/nvflare_workspace/server/transfer/<job_id>/
```

---

## Troubleshooting

### Container Won't Start

**Check logs:**
```bash
docker logs nvflare-server 2>&1 | tail -50
```

**Common issues:**

| Error | Cause | Solution |
|-------|-------|----------|
| Port already in use | Another process on 8002/8003 | `docker stop $(docker ps -q)` |
| Permission denied | User not in docker group | `sudo usermod -aG docker $USER && logout` |
| GPU not accessible | NVIDIA runtime not configured | See Prerequisites section |

### Clients Can't Connect to Server

**1. Verify server is listening:**
```bash
ss -tlnp | grep -E "8002|8003"
```

**2. Check firewall:**
```bash
sudo ufw status | grep -E "8002|8003"
```

**3. Test network connectivity:**
```bash
ansible -i ansible/inventory.ini nvflare_clients -m shell -a "nc -zv 172.19.1.7 8002"
```

**4. Verify /etc/hosts on clients:**
```bash
ansible -i ansible/inventory.ini nvflare_clients -a "grep server /etc/hosts"
```

### Data Not Found During Training

**Error:** `FileNotFoundError: Training data not found: /data/mnist/k3s-client-09_train.pt`

**Solution:** Distribute data before running jobs:
```bash
python scripts/distribute_splits.py --dataset mnist --split_method uniform
```

**Verify data exists:**
```bash
ansible -i ansible/inventory.ini all -a "ls ~/nvflare_data/mnist/"
```

### Image Build Fails

**Check disk space:**
```bash
ansible -i ansible/inventory.ini all -a "df -h /var/lib/docker"
```

**Clean old images:**
```bash
ansible -i ansible/inventory.ini all -a "docker system prune -af"
```

**Check network access to NVIDIA registry:**
```bash
docker pull nvcr.io/nvidia/pytorch:24.12-py3
```

### Full Network Diagnostics

Run the comprehensive diagnostics playbook:

```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/diagnose_network.yml
```

This checks:
- Container status and configuration
- UFW firewall rules
- Port availability
- Cross-node connectivity
- DNS resolution

---

## Updating Versions

### Updating NVFlare Version

1. **Update `ansible/Dockerfile`:**
   ```dockerfile
   RUN python3 -m pip install nvflare==2.8.0  # Change version
   ```

2. **Update provisioning venv version in `deploy_docker_nvflare.yml`:**
   ```yaml
   - name: Install NVFlare wheel into the virtual environment
     ansible.builtin.pip:
       name: nvflare==2.8.0  # Change version
   ```

3. **Remove old images and redeploy:**
   ```bash
   ansible -i ansible/inventory.ini all -a "docker rmi nvflare-pt-docker:latest"
   ansible-playbook -i ansible/inventory.ini ansible/playbooks/deploy_docker_nvflare.yml
   ```

### Updating PyTorch Container Version

1. **Check compatibility:**
   - Visit [NVIDIA PyTorch Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/)
   - Verify your GPU driver version meets the minimum requirement

2. **Update `ansible/Dockerfile`:**
   ```dockerfile
   ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.01-py3  # Change version
   ```

3. **Test locally first:**
   ```bash
   docker build -t nvflare-pt-docker:test -f ansible/Dockerfile ansible/
   docker run --rm --gpus all nvflare-pt-docker:test python -c "
     import torch
     import nvflare
     print(f'NVFlare: {nvflare.__version__}')
     print(f'PyTorch: {torch.__version__}')
     print(f'CUDA: {torch.cuda.is_available()}')
   "
   ```

4. **Deploy to cluster:**
   ```bash
   ansible -i ansible/inventory.ini all -a "docker rmi nvflare-pt-docker:latest"
   ansible-playbook -i ansible/inventory.ini ansible/playbooks/deploy_docker_nvflare.yml
   ```

### Version Compatibility Matrix

| NVFlare | PyTorch Container | PyTorch | CUDA | Min Driver | Status |
|---------|-------------------|---------|------|------------|--------|
| 2.7.1   | 24.12-py3        | 2.5.1   | 12.6 | 560.x      | Tested |
| 2.7.1   | 25.01-py3        | 2.6.0   | 12.8 | 570.x      | Tested |
| 2.7.0   | 25.10-py3        | 2.9.0a0 | 13.0 | 580.95     | Requires new driver |

---

## Quick Reference Commands

```bash
# === Deployment ===
ansible-playbook -i ansible/inventory.ini ansible/playbooks/setup_nvflare_networking.yml

# === Validation ===
ansible-playbook -i ansible/inventory.ini ansible/playbooks/validate_docker_gpu.yml
ansible-playbook -i ansible/inventory.ini ansible/playbooks/validate_nvflare_network.yml

# === Data Distribution ===
python scripts/distribute_splits.py --dataset mnist --split_method uniform
python scripts/distribute_splits.py --dataset cifar10 --split_method square

# === Job Operations ===
nvflare job submit -j jobs/configs/pt_mnist_fedavg
python scripts/job_status.py <job_id> --watch

# === Container Management ===
docker logs nvflare-server --tail 100
ansible -i ansible/inventory.ini all -a "docker ps"
ansible -i ansible/inventory.ini all -a "docker restart \$(docker ps -q)"

# === Diagnostics ===
ansible-playbook -i ansible/inventory.ini ansible/playbooks/diagnose_network.yml

# === GPU Monitoring ===
ansible -i ansible/inventory.ini all -a "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv"
```

---

## Support

- [NVFlare Documentation](https://nvflare.readthedocs.io/)
- [NVFlare GitHub Issues](https://github.com/NVIDIA/NVFlare/issues)
- [NVIDIA PyTorch Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
