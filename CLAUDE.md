# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Federated learning project built on NVIDIA FLARE (NVFlare) 2.7.1 for distributed machine learning across GPU-equipped client nodes. Uses Ansible for infrastructure management and Docker containers with host networking. Supports PyTorch models (MNIST, CIFAR-10, ResNet) with configurable non-IID data distribution.

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

### Validation Playbooks
```bash
# Test SSH connectivity
ansible-playbook -i ansible/inventory.ini ansible/playbooks/ping_all_nodes.yml

# Validate Docker and GPU prerequisites
ansible-playbook -i ansible/inventory.ini ansible/playbooks/validate_docker_gpu.yml

# Validate NVFlare network configuration
ansible-playbook -i ansible/inventory.ini ansible/playbooks/validate_nvflare_network.yml

# Full network diagnostics
ansible-playbook -i ansible/inventory.ini ansible/playbooks/diagnose_network.yml
```

### Individual Deployment Playbooks
```bash
# Deploy containers only (after manual provisioning)
ansible-playbook -i ansible/inventory.ini ansible/playbooks/deploy_docker_nvflare.yml

# Configure firewall (ports 8002/8003)
ansible-playbook -i ansible/inventory.ini ansible/playbooks/configure_ufw_nvflare.yml

# Configure /etc/hosts for hostname resolution
ansible-playbook -i ansible/inventory.ini ansible/playbooks/configure_hosts_file.yml
```

### Data Distribution
```bash
# MNIST with uniform (IID) split
python scripts/distribute_splits.py --dataset mnist --split_method uniform

# CIFAR-10 with non-IID (square) split
python scripts/distribute_splits.py --dataset cifar10 --split_method square

# Custom dataset
python scripts/distribute_splits.py --dataset custom --data_path /path/to/data.pt --split_method linear
```

Split methods: `uniform` (IID), `linear`, `square`, `exponential` (increasingly non-IID)

### Job Submission
```bash
# Submit pre-configured jobs
nvflare job submit -j jobs/configs/pt_mnist_fedavg
nvflare job submit -j jobs/configs/pt_cifar10_resnet18

# Generate custom job config
cd jobs/generators && python fedavg_mnist.py -p ../../project.yml
```

### Job Monitoring
```bash
# One-time status check
python scripts/job_status.py <job_id>

# Continuous monitoring with progress bars
python scripts/job_status.py <job_id> --watch

# Custom rounds/clients
python scripts/job_status.py <job_id> --watch --rounds 10 --clients 6
```

### Container Management
```bash
# View logs
docker logs nvflare-server
docker logs nvflare-client-09  # Pattern: nvflare-client-{node_number}

# View all container status
ansible -i ansible/inventory.ini all -a "docker ps"

# Restart all containers
ansible -i ansible/inventory.ini all -a "docker restart \$(docker ps -q)"
```

## Key Files

| File | Purpose |
|------|---------|
| `project.yml` | Defines participants (server, clients, admin) and Docker image name |
| `ansible/inventory.ini` | Node IPs, SSH users, NVFlare ports, data directories |
| `ansible/Dockerfile` | NVFlare container image (nvidia/pytorch:24.12-py3 + NVFlare 2.7.1) |
| `scripts/distribute_splits.py` | Dataset distribution with IID/non-IID splits |
| `scripts/job_status.py` | Real-time job monitoring with progress bars |

## Project Structure

```
federated_learning/
├── project.yml                 # NVFlare participant definitions
├── ansible/
│   ├── inventory.ini           # Node IPs and SSH configuration
│   ├── Dockerfile              # Container image definition
│   └── playbooks/              # Ansible automation
│       ├── setup_nvflare_networking.yml   # Master orchestration
│       ├── deploy_docker_nvflare.yml      # Container deployment
│       ├── validate_docker_gpu.yml        # GPU validation
│       └── ...
├── jobs/
│   ├── configs/                # Ready-to-submit job configurations
│   │   ├── pt_mnist_fedavg/    # MNIST + CNN example
│   │   └── pt_cifar10_resnet18/# CIFAR-10 + ResNet-18
│   ├── generators/             # Scripts to generate job configs
│   │   ├── fedavg_mnist.py
│   │   └── fedavg_cifar10.py
│   └── src/                    # Training scripts and models
│       ├── pt_mnist_fl.py      # MNIST training script
│       ├── pt_cifar10_fl.py    # CIFAR-10 training script
│       ├── mnist_cnn.py        # CNN model definition
│       └── resnet18_cifar10.py # ResNet-18 model definition
├── scripts/
│   ├── distribute_splits.py    # Dataset distribution
│   └── job_status.py           # Job monitoring
├── workspace/                  # Generated NVFlare workspace
│   └── example_project/prod_00/# Certificates and startup kits
└── docs/
    └── DEPLOYMENT.md           # Detailed deployment guide
```

## Job Structure

Jobs are stored in `jobs/configs/`. Each job has:
- `meta.json` - Job metadata and deployment map
- `app/config/` - Server and client configuration
- `app/custom/src/` - Training scripts and models

Training scripts use NVFlare Client API pattern:
```python
import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import ParamsType

flare.init()
while flare.is_running():
    input_model = flare.receive()
    # ... train ...
    output_model = flare.FLModel(
        params=model.state_dict(),
        params_type=ParamsType.FULL,
        metrics={"accuracy": accuracy}
    )
    flare.send(output_model)
```

## Development Workflow

1. **Configure participants** - Edit `project.yml` and `ansible/inventory.ini`
2. **Deploy cluster** - `ansible-playbook -i ansible/inventory.ini ansible/playbooks/setup_nvflare_networking.yml`
3. **Distribute data** - `python scripts/distribute_splits.py --dataset mnist --split_method uniform`
4. **Submit job** - `nvflare job submit -j jobs/configs/pt_mnist_fedavg`
5. **Monitor progress** - `python scripts/job_status.py <job_id> --watch`

## Adding/Removing Clients

1. Edit `ansible/inventory.ini` - add/remove client lines
2. Edit `project.yml` - add/remove participant entries
3. Re-deploy: `ansible-playbook -i ansible/inventory.ini ansible/playbooks/setup_nvflare_networking.yml`

## Important Notes

- **Never edit** auto-generated NVFlare files in `workspace/` - use Ansible playbooks instead
- **Data must be distributed** before running jobs - use `scripts/distribute_splits.py`
- Containers use `restart_policy: unless-stopped` (persist through reboots)
- Docker image: `nvflare-pt-docker` built from `nvcr.io/nvidia/pytorch:24.12-py3`
- NVFlare version: 2.7.1
- All playbooks use `community.docker.docker_container` module, not the generated `docker.sh` scripts
- Data directory: `~/nvflare_data/{dataset}/` mounted as `/data/` in containers
