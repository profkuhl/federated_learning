# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a federated learning infrastructure project that uses NVIDIA FLARE for distributed machine learning across multiple Kubernetes nodes. The project includes complete automation for deploying and managing federated learning clusters using Ansible.

## Key Commands

### Cluster Deployment
- **Deploy entire cluster**: `./deploy_flare_cluster.sh` (interactive script with options 1-7)
- **Deploy server only**: `cd ansible && ansible-playbook -i inventory.ini playbooks/deploy_flare_server.yml`
- **Deploy clients only**: `cd ansible && ansible-playbook -i inventory.ini playbooks/deploy_flare_clients.yml`
- **Deploy CIFAR-10 application**: `cd ansible && ansible-playbook -i inventory.ini playbooks/deploy_cifar10_app.yml`
- **Verify cluster**: `cd ansible && ansible-playbook -i inventory.ini playbooks/verify_flare_cluster.yml`
- **Install Docker**: `cd ansible && ansible-playbook -i inventory.ini playbooks/install_docker_ubuntu.yml`

### Service Management
- **Start FLARE server**: `sudo systemctl start nvflare-server`
- **Start FLARE clients**: `ansible -i ansible/inventory.ini flare_clients -b -m systemd -a "name=nvflare-client state=started"`
- **Check service status**: `sudo systemctl status nvflare-server` or `sudo systemctl status nvflare-client`
- **View logs**: `journalctl -u nvflare-server -f` or `tail -f /home/k3s-server-07/nvflare/logs/server.log`

### Job Management
- **Submit federated learning job**: `/home/k3s-server-07/nvflare/submit_job.sh`
- **Monitor job status**: `/home/k3s-server-07/nvflare/monitor_job.sh`
- **Access admin console**: `cd /home/k3s-server-07/nvflare/workspace/admin && ./start.sh`
- **Test federated learning**: `/home/k3s-server-07/nvflare/test_federated_learning.sh`

## Architecture

### Infrastructure Components
- **FLARE Server**: Runs on 192.168.1.7 (k3s-server-07) - coordinates federated learning
- **FLARE Clients**: Run on 192.168.1.6, 192.168.1.8, 192.168.1.9, 192.168.1.17, 192.168.1.18 - participate in training
- **Inventory**: `ansible/inventory.ini` defines server and client node mappings
- **Networking**: Uses ports 8002 (federated learning) and 8003 (admin console)

### Key Directories
- `/home/k3s-server-07/nvflare/`: Main FLARE installation directory
- `/home/k3s-server-07/nvflare/workspace/`: Contains server, client, and admin workspaces
- `/home/k3s-server-07/nvflare/jobs/`: Federated learning job definitions
- `/home/k3s-server-07/nvflare/logs/`: Service logs for debugging

### Federated Learning Application
- **Model**: SimpleCNN for CIFAR-10 classification (10 classes)
- **Data Distribution**: Non-IID data splits across clients (each client gets 4 out of 10 classes)
- **Training**: 5 rounds with 2 epochs per round, minimum 3 clients required
- **Aggregation**: Uses InTimeAccumulateWeightedAggregator for model averaging

### Ansible Playbooks
- `deploy_flare_server.yml`: Sets up FLARE server with systemd service
- `deploy_flare_clients.yml`: Deploys clients with Python 3.10 and FLARE 2.4.1
- `deploy_cifar10_app.yml`: Creates federated learning job with CIFAR-10 training
- `configure_networking.yml`: Sets up networking and firewall rules
- `verify_flare_cluster.yml`: Validates cluster deployment
- Infrastructure playbooks: GPU setup, NVIDIA drivers, containerd, DNS configuration

## Development Workflow

### Testing Changes
1. Make changes to playbooks or configurations
2. Run specific playbook: `ansible-playbook -i ansible/inventory.ini playbooks/<playbook>.yml`
3. Verify with: `ansible/playbooks/verify_flare_cluster.yml`
4. Test federated learning: Run option 7 in `deploy_flare_cluster.sh`

### Debugging
- Check service logs: `journalctl -u nvflare-server -f` or `journalctl -u nvflare-client -f`
- Monitor application logs: `tail -f /home/k3s-server-07/nvflare/logs/server.log`
- Run connectivity tests: `ansible -i ansible/inventory.ini all -m ping`
- Use monitoring script: `/home/k3s-server-07/nvflare/monitor_job.sh`

### Adding New Applications
1. Create job directory: `/home/k3s-server-07/nvflare/jobs/<app_name>/`
2. Structure: `app/config/`, `app/custom/`, `app/models/`
3. Define `meta.json` with job metadata
4. Create `config_fed_server.json` and `config_fed_client.json`
5. Implement training executor in `app/custom/`
6. Generate initial model in `app/models/`

## Important Notes

- All clients use Python 3.10 installed via uv package manager
- NVIDIA FLARE version 2.4.1 is installed with cryptography<42 constraint
- Server runs on local connection, clients use SSH with key-based authentication
- Data is automatically downloaded to `/home/k3s-server-07/nvflare/data/` on each client
- Each client gets assigned a site name (site-1, site-2, etc.) based on inventory order
- Jobs are submitted through the admin console interface
- The cluster supports GPU acceleration when available (CUDA detection in training code)