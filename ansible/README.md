# Ansible Playbooks for NVFlare

This directory contains Ansible playbooks and configuration files for deploying and managing a dockerized NVFlare federated learning cluster.

## Quick Start

```bash
# Full deployment (recommended - runs all steps in order)
ansible-playbook -i ansible/inventory.ini ansible/playbooks/setup_nvflare_networking.yml
```

## Directory Structure

```
ansible/
├── inventory.ini               # Node IPs, SSH users, and configuration variables
├── Dockerfile                  # NVFlare container image definition
├── README.md                   # This file
└── playbooks/
    ├── setup_nvflare_networking.yml    # Master orchestration playbook
    ├── deploy_docker_nvflare.yml       # Main deployment playbook
    ├── configure_ufw_nvflare.yml       # Firewall configuration
    ├── configure_hosts_file.yml        # Hostname resolution
    ├── validate_docker_gpu.yml         # GPU prerequisites validation
    ├── validate_nvflare_network.yml    # Post-deployment validation
    ├── diagnose_network.yml            # Network diagnostics
    ├── ping_all_nodes.yml              # Basic connectivity test
    └── install_nvidia_drivers.yml      # NVIDIA driver installation
```

---

## Configuration Files

### inventory.ini

Defines cluster nodes and configuration variables:

```ini
[nvflare_server]
172.19.1.7  ansible_user=k3s-server-07 ansible_connection=local

[nvflare_clients]
k3s-client-08 ansible_host=172.19.1.8  ansible_user=k3s-client-08
k3s-client-09 ansible_host=172.19.1.9  ansible_user=k3s-client-09

[all:vars]
nvflare_server_ip=172.19.1.7
nvflare_server_port=8002
nvflare_admin_port=8003
nvflare_workspace_dir=/home/{{ ansible_user }}/nvflare_workspace
nvflare_data_dir=/home/{{ ansible_user }}/nvflare_data
```

**Key variables:**
- `nvflare_server_ip` - Server IP address (used by clients for connection)
- `nvflare_workspace_dir` - Where startup kits are stored on each node
- `nvflare_data_dir` - Where training data is stored (mounted as `/data` in containers)

### Dockerfile

Defines the NVFlare container image:

```dockerfile
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3
FROM ${BASE_IMAGE}

RUN python3 -m pip install nvflare==2.7.1 torch torchvision numpy pandas polars

WORKDIR /workspace/
```

---

## Playbooks

### Master Orchestration

#### setup_nvflare_networking.yml

**Purpose**: Complete NVFlare deployment in one command. Orchestrates all other playbooks in the correct order.

**What it does:**
1. Configures UFW firewall rules
2. Configures /etc/hosts for hostname resolution
3. Deploys NVFlare containers (builds images, distributes startup kits, starts containers)
4. Validates the deployment

**Usage:**
```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/setup_nvflare_networking.yml
```

**When to use:** First-time deployment or complete redeployment after configuration changes.

---

### Deployment Playbooks

#### deploy_docker_nvflare.yml

**Purpose**: Main deployment playbook that provisions NVFlare workspace, builds Docker images, distributes startup kits, and starts containers.

**What it does on the server:**
1. Creates provisioning virtualenv (`/opt/nvflare_provision_venv`)
2. Runs `nvflare provision` to generate workspace with certificates
3. Builds Docker image `nvflare-pt-docker:latest`
4. Starts `nvflare-server` container
5. Starts `nvflare-client-local` container (for local client on server)

**What it does on clients:**
1. Builds Docker image (skips if already exists with correct version)
2. Receives startup kit via rsync from server
3. Starts `nvflare-client-XX` container

**Usage:**
```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/deploy_docker_nvflare.yml
```

**When to use:** After making changes to `project.yml`, `inventory.ini`, or `Dockerfile`.

---

### Network Configuration Playbooks

#### configure_ufw_nvflare.yml

**Purpose**: Configure firewall rules to allow NVFlare communication.

**What it does:**
- Ensures UFW is installed and enabled
- Allows SSH (port 22) to prevent lockout
- Allows ports 8002/8003 from the internal subnet
- Allows all traffic from the NVFlare subnet (for Docker host networking)

**Ports opened:**
| Port | Protocol | Purpose |
|------|----------|---------|
| 22 | TCP | SSH access |
| 8002 | TCP | Federated learning communication |
| 8003 | TCP | Admin console |

**Usage:**
```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/configure_ufw_nvflare.yml
```

**When to use:** Before deployment or when firewall rules are reset.

---

#### configure_hosts_file.yml

**Purpose**: Configure `/etc/hosts` on all nodes for consistent hostname resolution.

**What it does:**
- Adds `server` → server IP mapping
- Adds `mylocalhost` → server IP mapping (required by NVFlare)
- Adds all client hostnames → their IPs
- Verifies hostname resolution works

**Usage:**
```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/configure_hosts_file.yml
```

**When to use:** When adding new clients or if hostname resolution fails.

---

### Validation Playbooks

#### validate_docker_gpu.yml

**Purpose**: Verify Docker and GPU prerequisites before deployment.

**What it checks:**
- Docker is installed and running
- `nvidia-smi` is available
- NVIDIA Container Toolkit is installed
- Docker can run GPU workloads

**Usage:**
```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/validate_docker_gpu.yml
```

**When to use:** Before first deployment to ensure all prerequisites are met.

**Example output:**
```
Node: k3s-client-09
Status: READY for NVFlare deployment
- Docker: Available
- GPU: Available
- Docker GPU Access: Verified
```

---

#### validate_nvflare_network.yml

**Purpose**: Validate NVFlare deployment after containers are running.

**What it checks:**
- All containers are running
- Containers use host networking
- Restart policy is `unless-stopped`
- Ports 8002/8003 are listening
- Network connectivity between nodes

**Usage:**
```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/validate_nvflare_network.yml
```

**When to use:** After deployment to verify everything is working.

---

#### diagnose_network.yml

**Purpose**: Comprehensive network diagnostics for troubleshooting.

**What it checks:**
- Docker version and running containers
- Container network mode
- UFW firewall status and rules
- Listening ports (8002/8003)
- Server-to-client connectivity
- Client-to-server connectivity

**Usage:**
```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/diagnose_network.yml
```

**When to use:** When clients can't connect or jobs fail to start.

---

### Utility Playbooks

#### ping_all_nodes.yml

**Purpose**: Simple connectivity test to verify SSH access to all nodes.

**Usage:**
```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/ping_all_nodes.yml
```

**When to use:** First step to verify Ansible can reach all nodes.

---

#### install_nvidia_drivers.yml

**Purpose**: Install NVIDIA drivers on nodes that don't have them.

**What it does:**
- Detects Ubuntu version
- Installs appropriate NVIDIA driver package
- Reboots if necessary

**Usage:**
```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/install_nvidia_drivers.yml
```

**When to use:** On new nodes without NVIDIA drivers installed.

---

## Common Operations

### View Container Status Across Cluster

```bash
ansible -i ansible/inventory.ini all -a "docker ps --format 'table {{.Names}}\t{{.Status}}'"
```

### View Container Logs

```bash
# Server logs
ansible -i ansible/inventory.ini nvflare_server -a "docker logs nvflare-server --tail 50"

# Specific client logs
ansible -i ansible/inventory.ini k3s-client-09 -a "docker logs nvflare-client-09 --tail 50"

# All client logs
ansible -i ansible/inventory.ini nvflare_clients -m shell -a "docker logs \$(docker ps -q) --tail 20"
```

### Restart All Containers

```bash
ansible -i ansible/inventory.ini all -a "docker restart \$(docker ps -q)"
```

### Stop All Containers

```bash
ansible -i ansible/inventory.ini all -a "docker stop \$(docker ps -q)"
```

### Remove All Containers (for clean redeployment)

```bash
ansible -i ansible/inventory.ini all -m shell -a "docker rm -f \$(docker ps -aq) || true"
```

### Check GPU Usage

```bash
ansible -i ansible/inventory.ini all -a "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv"
```

### Check Disk Space

```bash
ansible -i ansible/inventory.ini all -a "df -h /var/lib/docker"
```

### Clean Docker Resources

```bash
# Remove unused images, containers, networks
ansible -i ansible/inventory.ini all -a "docker system prune -af"

# Also remove volumes (caution: may delete data)
ansible -i ansible/inventory.ini all -a "docker system prune -af --volumes"
```

---

## Targeting Specific Hosts

### Run on Server Only

```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/deploy_docker_nvflare.yml --limit nvflare_server
```

### Run on Specific Client

```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/validate_docker_gpu.yml --limit k3s-client-09
```

### Run on All Clients

```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/validate_docker_gpu.yml --limit nvflare_clients
```

---

## Adding a New Client

1. **Edit `inventory.ini`:**
   ```ini
   [nvflare_clients]
   k3s-client-09 ansible_host=172.19.1.9  ansible_user=k3s-client-09
   k3s-client-10 ansible_host=172.19.1.10 ansible_user=k3s-client-10  # New client
   ```

2. **Edit `project.yml`:**
   ```yaml
   participants:
     # ... existing clients ...
     - name: k3s-client-10
       type: client
       org: nvidia
   ```

3. **Verify prerequisites on new client:**
   ```bash
   ansible-playbook -i ansible/inventory.ini ansible/playbooks/validate_docker_gpu.yml --limit k3s-client-10
   ```

4. **Run full deployment:**
   ```bash
   ansible-playbook -i ansible/inventory.ini ansible/playbooks/setup_nvflare_networking.yml
   ```

---

## Troubleshooting

### "Permission denied" on Docker commands

User not in docker group:
```bash
ansible -i ansible/inventory.ini <host> -b -a "usermod -aG docker <username>"
```
Then logout/login on that host.

### SSH connection refused

Verify SSH access manually:
```bash
ssh k3s-client-09@172.19.1.9
```

Check SSH is running:
```bash
ansible -i ansible/inventory.ini <host> -a "systemctl status sshd"
```

### Docker image build fails

Check disk space:
```bash
ansible -i ansible/inventory.ini <host> -a "df -h /"
```

Check network access to NVIDIA registry:
```bash
ansible -i ansible/inventory.ini <host> -a "docker pull nvcr.io/nvidia/pytorch:24.12-py3"
```

### Firewall blocking connections

Check UFW status:
```bash
ansible -i ansible/inventory.ini <host> -b -a "ufw status verbose"
```

Reset and reconfigure:
```bash
ansible-playbook -i ansible/inventory.ini ansible/playbooks/configure_ufw_nvflare.yml
```

---

## Version Information

| Component | Version |
|-----------|---------|
| NVFlare | 2.7.1 |
| Base Image | nvcr.io/nvidia/pytorch:24.12-py3 |
| PyTorch | 2.5.1 |
| CUDA | 12.6 |
| Minimum NVIDIA Driver | 560.x |
