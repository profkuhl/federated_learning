# Federated Learning with NVIDIA FLARE

A production-ready federated learning cluster built on NVIDIA FLARE (NVFlare) 2.7.1. Supports distributed machine learning across GPU-equipped nodes with PyTorch models, configurable non-IID data distribution, and comprehensive infrastructure automation via Ansible.

## Features

- **Multi-node GPU Cluster**: Deploy across multiple physical machines with NVIDIA GPUs
- **Docker-based Deployment**: Consistent containerized environment with NVIDIA PyTorch base image
- **Ansible Automation**: One-command deployment, validation, and diagnostics
- **Non-IID Data Support**: Configurable data splits (uniform, linear, square, exponential)
- **Pre-built Examples**: MNIST and CIFAR-10 with FedAvg algorithm
- **Job Monitoring**: Real-time progress visualization with graphical output

## Quick Start

### Prerequisites

- Ubuntu 22.04+ on all nodes
- NVIDIA GPU with driver 570.x+ on each node
- Docker with NVIDIA Container Toolkit
- SSH access between nodes
- Ansible 2.10+

### 1. Clone and Configure

```bash
git clone <repository_url>
cd federated_learning

# Edit participant list (server + clients)
vim project.yml

# Edit node IPs and SSH users
vim ansible/inventory.ini
```

### 2. Validate Prerequisites

```bash
# Test connectivity to all nodes
ansible-playbook -i ansible/inventory.ini ansible/playbooks/ping_all_nodes.yml

# Verify Docker and GPU access
ansible-playbook -i ansible/inventory.ini ansible/playbooks/validate_docker_gpu.yml
```

### 3. Deploy the Cluster

```bash
# Full deployment: UFW rules, /etc/hosts, Docker containers, validation
ansible-playbook -i ansible/inventory.ini ansible/playbooks/setup_nvflare_networking.yml
```

This command:
1. Configures firewall rules (ports 8002/8003)
2. Sets up hostname resolution
3. Provisions NVFlare workspace with certificates
4. Builds Docker images on all nodes
5. Distributes startup kits to clients
6. Starts server and client containers
7. Validates the deployment

### 4. Distribute Training Data

```bash
# MNIST with uniform (IID) split
python scripts/distribute_splits.py --dataset mnist --split_method uniform

# CIFAR-10 with non-IID (square) split
python scripts/distribute_splits.py --dataset cifar10 --split_method square
```

### 5. Submit a Training Job

```bash
# Submit the pre-configured MNIST job
nvflare job submit -j jobs/configs/pt_mnist_fedavg

# Or the GPU-intensive CIFAR-10 job
nvflare job submit -j jobs/configs/pt_cifar10_resnet18
```

### 6. Monitor Progress

```bash
# One-time status check
python scripts/job_status.py <job_id>

# Continuous monitoring
python scripts/job_status.py <job_id> --watch
```

## Project Structure

```
federated_learning/
├── project.yml                 # NVFlare participant definitions
├── ansible/
│   ├── inventory.ini           # Node IPs and SSH configuration
│   ├── Dockerfile              # NVFlare container image definition
│   └── playbooks/              # Ansible automation playbooks
├── jobs/
│   ├── configs/                # Ready-to-submit job configurations
│   │   ├── pt_mnist_fedavg/    # MNIST + CNN (simple example)
│   │   └── pt_cifar10_resnet18/# CIFAR-10 + ResNet-18 (GPU-intensive)
│   ├── generators/             # Scripts to generate job configs
│   └── src/                    # Training scripts and models
├── scripts/
│   ├── distribute_splits.py    # Dataset distribution to clients
│   └── job_status.py           # Job monitoring with progress bars
├── workspace/                  # Generated NVFlare workspace (certificates, startup kits)
└── docs/
    └── DEPLOYMENT.md           # Detailed deployment guide
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Complete deployment runbook with troubleshooting |
| [jobs/README.md](jobs/README.md) | Job creation and submission guide |
| [CLAUDE.md](CLAUDE.md) | Quick reference for AI assistants |

## Architecture

```
                    ┌─────────────────────────┐
                    │   NVFlare Server        │
                    │   172.19.1.7            │
                    │   Ports: 8002, 8003     │
                    └───────────┬─────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌───────────────┐       ┌───────────────┐
│ Client Node 1 │     │ Client Node 2 │  ...  │ Client Node N │
│ (GPU)         │     │ (GPU)         │       │ (GPU)         │
└───────────────┘     └───────────────┘       └───────────────┘
```

- **Server**: Coordinates training, aggregates model updates (FedAvg)
- **Clients**: Train on local data, send model updates to server
- **Communication**: Secure gRPC over host networking

## Version Information

| Component | Version |
|-----------|---------|
| NVFlare | 2.7.1 |
| PyTorch | 2.5.1 (from nvidia/pytorch:24.12-py3) |
| CUDA | 12.6 |
| Base Image | nvcr.io/nvidia/pytorch:24.12-py3 |

## Common Operations

```bash
# View container logs
docker logs nvflare-server
docker logs nvflare-client-09

# Restart all containers
ansible -i ansible/inventory.ini all -a "docker restart \$(docker ps -q)"

# Network diagnostics
ansible-playbook -i ansible/inventory.ini ansible/playbooks/diagnose_network.yml

# Check GPU usage across cluster
ansible -i ansible/inventory.ini all -a "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv"
```

## Adding/Removing Clients

1. Edit `ansible/inventory.ini` - add/remove client entries
2. Edit `project.yml` - add/remove participant entries
3. Re-deploy: `ansible-playbook -i ansible/inventory.ini ansible/playbooks/setup_nvflare_networking.yml`

## License

Apache License 2.0

## Resources

- [NVFlare Documentation](https://nvflare.readthedocs.io/)
- [NVFlare GitHub](https://github.com/NVIDIA/NVFlare)
- [NVIDIA PyTorch Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
