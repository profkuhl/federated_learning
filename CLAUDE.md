# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Federated learning project built on NVIDIA FLARE (NVFlare) for distributed machine learning across GPU-equipped client nodes. Uses Ansible for infrastructure management and Docker containers with host networking. Supports PyTorch models (MNIST, CIFAR-10, ResNet) with configurable non-IID data distribution.

**Detailed deployment guide**: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

## Architecture

**Three-Tier Structure:**
1. **Infrastructure (Ansible)** - Provisions and deploys NVFlare server/clients via Docker
2. **NVFlare Layer** - Federated learning framework handling server-client communication
3. **Jobs (Application)** - Training scripts, model definitions, and job configurations

**Network Topology:**
- Server at `172.19.1.7` (ports 8002 fed learn, 8003 admin)
- Clients in `172.19.1.x` subnet with host networking (no port mapping)
- Participants defined in `project.yml`, IPs in `ansible/inventory.ini`

## Common Commands

### Full Deployment (Recommended)
```bash
# Complete NVFlare setup: UFW rules, /etc/hosts, containers, validation
ansible-playbook -i ansible/inventory.ini ansible/playbooks/setup_nvflare_networking.yml
```

### Individual Playbooks
```bash
# Test connectivity
ansible-playbook -i ansible/inventory.ini ansible/playbooks/ping_all_nodes.yml

# Deploy containers only
ansible-playbook -i ansible/inventory.ini ansible/playbooks/deploy_docker_nvflare.yml

# Network diagnostics
ansible-playbook -i ansible/inventory.ini ansible/playbooks/diagnose_network.yml

# Configure firewall (ports 8002/8003)
ansible-playbook -i ansible/inventory.ini ansible/playbooks/configure_ufw_nvflare.yml
```

### NVFlare Operations
```bash
# Provision workspace (generates certs and startup kits)
nvflare provision -p project.yml -w workspace/example_project

# Generate job configuration from project.yml
cd jobs/pt_resnet && python fedavg_script_runner_pt.py -p ../../project.yml
```

### Container Management
```bash
# View logs
docker logs nvflare-server
docker logs nvflare-client-08  # Pattern: nvflare-client-{node_number}

# Stop/restart via Ansible
ansible -i ansible/inventory.ini all -a "docker stop \$(docker ps -q)"
ansible -i ansible/inventory.ini all -a "docker restart \$(docker ps -q)"
```

### Data Distribution
```bash
# Split and distribute dataset to clients
python scripts/distribute_splits.py \
  --data_path /path/to/dataset.csv \
  --split_method square \
  --inventory ansible/inventory.ini \
  --remote_dest /tmp/nvflare/data_splits
```

Split methods: `uniform` (IID), `linear`, `square`, `exponential` (increasingly heterogeneous)

## Key Files

| File | Purpose |
|------|---------|
| `project.yml` | Defines participants (server, clients, admin) and Docker image name |
| `ansible/inventory.ini` | Node IPs, SSH users, NVFlare ports |
| `ansible/Dockerfile` | NVFlare container image (NVIDIA PyTorch base + NVFlare 2.7.0) |
| `notebooks/utils/data_utils.py` | `split_and_distribute()` for tensor datasets |
| `scripts/distribute_splits.py` | CLI for file-based dataset splitting |

## Job Structure

Jobs under `jobs/pt_resnet/`:
- `fedavg_script_runner_pt.py` - Reads `project.yml`, generates per-client configs
- `src/pt_mnist_fl.py` - Training script using NVFlare Client API
- `src/mnist_cnn.py`, `src/resnet_18.py` - Model definitions
- `job_config/` - Generated output with `app_server/` and `app_<client>/` directories

Training scripts must use NVFlare Client API pattern:
```python
flare.init()
while flare.is_running():
    model = flare.receive()
    # ... train ...
    flare.send(FLModel(params=model.state_dict()))
```

## Development Workflow

1. Update `project.yml` with participants and `ansible/inventory.ini` with IPs
2. Provision: `nvflare provision -p project.yml -w workspace/example_project`
3. Deploy: `ansible-playbook -i ansible/inventory.ini ansible/playbooks/setup_nvflare_networking.yml`
4. Distribute data using `data_utils.py` or `distribute_splits.py`
5. Generate job: `cd jobs/pt_resnet && python fedavg_script_runner_pt.py -p ../../project.yml`
6. Submit job via NVFlare admin console

## Adding/Removing Clients

1. Edit `ansible/inventory.ini` (uncomment/add client lines)
2. Edit `project.yml` (add/remove participant entries)
3. Re-provision: `nvflare provision -p project.yml -w workspace/example_project`
4. Re-deploy: `ansible-playbook -i ansible/inventory.ini ansible/playbooks/setup_nvflare_networking.yml`

## Important Notes

- **Never edit** auto-generated NVFlare files in `workspace/` - use Ansible playbooks instead
- **Data is not auto-distributed** - must manually split and copy before running jobs
- Containers use `restart_policy: unless-stopped` (persist through reboots)
- Docker image: `nvflare-pt-docker` built from `nvcr.io/nvidia/pytorch:25.10-py3`
- All playbooks use `community.docker.docker_container` module, not the generated `docker.sh` scripts
