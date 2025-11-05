# NVFlare Distributed Deployment Runbook

This guide walks through deploying a dockerized NVFlare federated learning cluster across multiple physical machines.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Deployment Steps](#deployment-steps)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Updating Versions](#updating-versions)

---

## Prerequisites

### 1. Verify Cluster Connectivity

Test SSH access to all nodes:

```bash
cd ansible
ansible -i inventory.ini all -m ping
```

**Expected**: All nodes should return `pong`.

### 2. Verify Docker Installation

Check Docker is installed on all nodes:

```bash
ansible -i inventory.ini all -a "docker --version"
```

**Expected**: Docker version output from all 6 nodes (1 server + 5 clients).

### 3. Verify NVIDIA GPU Access

Check GPUs are available on all nodes:

```bash
ansible -i inventory.ini all -a "nvidia-smi"
```

**Expected**: GPU information from all nodes. If this fails, install NVIDIA drivers first:

```bash
ansible-playbook -i inventory.ini playbooks/install_nvidia_drivers.yml
```

### 4. Verify NVIDIA Docker Runtime

Check nvidia-docker runtime is installed:

```bash
ansible -i inventory.ini all -a "docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi"
```

**Expected**: nvidia-smi output from within a container on all nodes.

**If this fails**, install nvidia-docker2:
```bash
ansible -i inventory.ini all -b -m shell -a "
  distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID) && \\
  curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add - && \\
  curl -s -L https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list | \\
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \\
  apt-get update && apt-get install -y nvidia-docker2 && \\
  systemctl restart docker
"
```

---

## Deployment Steps

### Step 1: Clean Up Old Deployment (if exists)

If you have previous containers running, stop them:

```bash
# Stop server
ssh k3s-server-07@172.19.1.7 "docker stop flserver && docker rm flserver"

# Stop all clients
ansible -i ansible/inventory.ini nvflare_clients -a "docker stop \$(docker ps -q) || true"
ansible -i ansible/inventory.ini nvflare_clients -a "docker rm \$(docker ps -aq) || true"
```

### Step 2: Update Configuration Files

Verify your configuration is correct:

**Check `project.yml`:**
```bash
grep "docker_image:" project.yml
```
**Expected**: `docker_image: nvflare-pt-docker` (uncommented)

**Check `ansible/Dockerfile`:**
```bash
head -n 15 ansible/Dockerfile
```
**Expected**:
- Base image: `nvcr.io/nvidia/pytorch:25.10-py3` (or your preferred version)
- NVFlare version: `nvflare==2.7.0` (or your preferred version)

**Check `ansible/inventory.ini`:**
```bash
cat ansible/inventory.ini
```
**Expected**: All 5 client nodes listed under `[nvflare_clients]`, server under `[nvflare_server]`

### Step 3: Re-Provision NVFlare Workspace

This creates fresh certificates and startup kits with Docker support:

```bash
# Backup old workspace (optional)
mv workspace/example_project workspace/example_project.backup.$(date +%Y%m%d_%H%M%S)

# Provision new workspace
nvflare provision -p project.yml -w workspace/example_project
```

**Verify** docker.sh scripts were generated:
```bash
ls workspace/example_project/prod_00/172.19.1.7/startup/docker.sh
ls workspace/example_project/prod_00/k3s-client-09/startup/docker.sh
```

Both files should exist.

### Step 4: Deploy to All Nodes

This will:
- Build Docker images on all nodes
- Distribute startup kits
- Configure GPU access and auto-restart
- Start all containers

```bash
cd ansible
ansible-playbook -i inventory.ini playbooks/deploy_docker_nvflare.yml
```

**Expected Duration**: 15-30 minutes (depends on network speed and build time)

**Watch for**:
- "Build the custom NVFlare Docker image" tasks (will take 10+ minutes each)
- "Start the server container" - should complete without errors
- "Start the client container" - should complete on all 5 clients

### Step 5: Verify Deployment

See [Verification](#verification) section below.

---

## Verification

### 1. Check All Containers Are Running

**Server:**
```bash
docker ps
```
**Expected**: 1 container named `flserver`, status `Up`

**Clients:**
```bash
ansible -i ansible/inventory.ini nvflare_clients -a "docker ps"
```
**Expected**: Each client shows 1 container:
- k3s-client-06: `nvflare-client-06`
- k3s-client-08: `nvflare-client-08`
- k3s-client-09: `nvflare-client-09`
- k3s-client-17: `nvflare-client-17`
- k3s-client-18: `nvflare-client-18`

### 2. Check Container Logs

**Server:**
```bash
docker logs flserver --tail 50
```
**Look for**: `Server started` or `Listening on port 8002`

**Clients:**
```bash
ansible -i ansible/inventory.ini k3s-client-09 -a "docker logs nvflare-client-09 --tail 50"
```
**Look for**: `Connected to server` or `Registered with server`

### 3. Verify GPU Access

**From server:**
```bash
docker exec flserver nvidia-smi
```

**From all clients:**
```bash
ansible -i ansible/inventory.ini nvflare_clients -m shell -a "
  hostname_num=\$(hostname | grep -oP 'client-\K[0-9]+');
  docker exec nvflare-client-\$hostname_num nvidia-smi
"
```

**Expected**: GPU information displayed from all containers.

### 4. Verify Restart Policy

```bash
ansible -i ansible/inventory.ini all -m shell -a "
  docker inspect --format='Container: {{.Name}}, Restart: {{.HostConfig.RestartPolicy.Name}}' \$(docker ps -q)
"
```

**Expected**: All containers show `Restart: unless-stopped`

<!-- ### 5. Check Network Connectivity

**From a client to server:**
```bash
ansible -i ansible/inventory.ini k3s-client-09 -m shell -a "
  docker exec nvflare-client-09 ping -c 3 172.19.1.7
"
```

**Expected**: Successful pings. -->

---

## Troubleshooting

### Container Won't Start

**Check logs:**
```bash
docker logs flserver  # or nvflare-client-XX
```

**Common issues:**
- **Port already in use**: Another process using 8002/8003
  - Fix: `docker stop $(docker ps -q)` or `netstat -tulpn | grep 8002`
- **Permission denied**: User not in docker group
  - Fix: `sudo usermod -aG docker $USER` then logout/login
- **GPU not accessible**: NVIDIA runtime not configured
  - Fix: See Prerequisites section

### Clients Can't Connect to Server

**Verify server IP in client config:**
```bash
ansible -i ansible/inventory.ini k3s-client-09 -a "
  grep 172.19.1.7 /home/k3s-client-09/nvflare_workspace/k3s-client-09/startup/fed_client.json
"
```

**Check firewall:**
```bash
# On server
sudo ufw status
sudo ufw allow 8002/tcp
sudo ufw allow 8003/tcp
```

### Container Keeps Restarting

**Check logs for crash reason:**
```bash
docker logs nvflare-client-09 --tail 100
```

**Common causes:**
- Missing data files
- Incorrect certificates
- Out of memory

**To stop auto-restart temporarily:**
```bash
docker update --restart=no nvflare-client-09
```

### Image Build Fails

**Check Docker disk space:**
```bash
ansible -i ansible/inventory.ini all -a "df -h /var/lib/docker"
```

**Clean old images:**
```bash
ansible -i ansible/inventory.ini all -a "docker system prune -af"
```

### Data Not Found During Training

Make sure data is distributed to clients **before** running jobs:

```bash
# Example for MNIST
cd notebooks
jupyter notebook GatherDataset.ipynb
# Run the cells to distribute data

# Verify data exists on clients
ansible -i ansible/inventory.ini nvflare_clients -a "ls -lh /tmp/mnist_data/"
```

---

## Updating Versions

### Updating NVFlare Version

1. **Check available versions:**
   ```bash
   pip index versions nvflare
   ```

2. **Update `ansible/Dockerfile`:**
   ```dockerfile
   # Change this line:
   RUN python3 -m pip install nvflare==2.7.0

   # To (example):
   RUN python3 -m pip install nvflare==2.8.0
   ```

3. **Update `jobs/pt_resnet/requirements.txt`:**
   ```
   nvflare~=2.8.0
   torch
   torchvision
   ```

4. **Rebuild and redeploy:**
   ```bash
   # Remove old images on all nodes
   ansible -i ansible/inventory.ini all -a "docker rmi nvflare-pt-docker:latest"

   # Re-run deployment
   cd ansible
   ansible-playbook -i inventory.ini playbooks/deploy_docker_nvflare.yml
   ```

### Updating PyTorch Container Version

1. **Find latest NVIDIA PyTorch container:**
   - Visit: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
   - Or check release notes: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/

2. **Update `ansible/Dockerfile`:**
   ```dockerfile
   # Change this line:
   ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3

   # To (example for November 2025):
   ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
   ```

3. **Check CUDA compatibility:**
   Each container version includes specific CUDA/cuDNN versions. Verify compatibility:
   ```bash
   docker run --rm nvcr.io/nvidia/pytorch:25.11-py3 python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
   ```

4. **Rebuild images:**
   ```bash
   # Remove old images
   ansible -i ansible/inventory.ini all -a "docker rmi nvflare-pt-docker:latest"
   ansible -i ansible/inventory.ini all -a "docker rmi nvcr.io/nvidia/pytorch:25.10-py3"

   # Re-run deployment
   cd ansible
   ansible-playbook -i inventory.ini playbooks/deploy_docker_nvflare.yml
   ```

### Updating Both NVFlare and PyTorch

If updating both simultaneously:

1. **Update `ansible/Dockerfile`** with both changes
2. **Verify compatibility** between NVFlare and PyTorch versions
   - Check NVFlare release notes for supported PyTorch versions
   - URL: https://github.com/NVIDIA/NVFlare/releases

3. **Test locally first** (optional but recommended):
   ```bash
   cd ansible
   docker build -t nvflare-pt-docker:test -f Dockerfile .
   docker run --rm --gpus all nvflare-pt-docker:test python -c "
     import torch
     import nvflare
     print(f'NVFlare: {nvflare.__version__}')
     print(f'PyTorch: {torch.__version__}')
     print(f'CUDA Available: {torch.cuda.is_available()}')
   "
   ```

4. **If test succeeds, deploy to cluster:**
   ```bash
   # Clean old images
   ansible -i ansible/inventory.ini all -a "docker system prune -af --volumes"

   # Deploy
   ansible-playbook -i inventory.ini playbooks/deploy_docker_nvflare.yml
   ```

### Version Compatibility Matrix

Keep track of tested combinations:

| NVFlare | PyTorch Container | PyTorch | CUDA | Status |
|---------|-------------------|---------|------|--------|
| 2.7.0   | 25.10-py3        | 2.9.0a0 | 13.0 | ✅ Working |
| 2.6.2   | 25.06-py3        | 2.8.0a0 | 12.9 | ✅ Working |

Add your own tested versions here for future reference.

---

## Quick Reference Commands

```bash
# View all containers
ansible -i ansible/inventory.ini all -a "docker ps"

# View logs from all containers
ansible -i ansible/inventory.ini all -m shell -a "docker logs \$(docker ps -q) --tail 20"

# Restart all containers
ansible -i ansible/inventory.ini all -a "docker restart \$(docker ps -q)"

# Stop all containers
ansible -i ansible/inventory.ini all -a "docker stop \$(docker ps -q)"

# Remove all containers and start fresh
ansible -i ansible/inventory.ini all -a "docker rm -f \$(docker ps -aq)"
cd ansible && ansible-playbook -i inventory.ini playbooks/deploy_docker_nvflare.yml

# Check GPU usage across cluster
ansible -i ansible/inventory.ini all -a "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv"
```

---

## Support

For issues:
- NVFlare Documentation: https://nvflare.readthedocs.io/
- NVFlare GitHub Issues: https://github.com/NVIDIA/NVFlare/issues
- NVIDIA PyTorch Container: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
