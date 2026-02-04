# NVFlare Jobs

This directory contains federated learning job configurations, generators, and training scripts for NVIDIA FLARE.

## Directory Structure

```
jobs/
├── configs/                    # Ready-to-submit job configurations
│   ├── pt_mnist_fedavg/        # MNIST + CNN (simple, quick)
│   └── pt_cifar10_resnet18/    # CIFAR-10 + ResNet-18 (GPU-intensive)
├── generators/                 # Scripts to generate job configs
│   ├── fedavg_mnist.py         # Generates MNIST FedAvg job
│   └── fedavg_cifar10.py       # Generates CIFAR-10 FedAvg job
└── src/                        # Training scripts and model definitions
    ├── pt_mnist_fl.py          # MNIST training script
    ├── pt_cifar10_fl.py        # CIFAR-10 training script
    ├── mnist_cnn.py            # Simple CNN for MNIST
    ├── resnet18_cifar10.py     # ResNet-18 adapted for CIFAR-10
    └── resnet_18.py            # Standard ResNet-18 wrapper
```

## Quick Start

### 1. Distribute Data First

Before running any job, distribute the dataset to all clients:

```bash
# MNIST (simple, ~10MB per client)
python scripts/distribute_splits.py --dataset mnist --split_method uniform

# CIFAR-10 (larger, ~30MB per client)
python scripts/distribute_splits.py --dataset cifar10 --split_method square
```

### 2. Submit a Pre-configured Job

```bash
# Simple MNIST job (5 rounds, ~2-3 minutes)
nvflare job submit -j jobs/configs/pt_mnist_fedavg

# GPU-intensive CIFAR-10 job (10 rounds, ~10-15 minutes)
nvflare job submit -j jobs/configs/pt_cifar10_resnet18
```

### 3. Monitor Progress

```bash
python scripts/job_status.py <job_id> --watch
```

---

## Pre-configured Jobs

### MNIST FedAvg (`configs/pt_mnist_fedavg/`)

A simple federated learning job for testing and validation.

| Parameter | Value |
|-----------|-------|
| Dataset | MNIST (28x28 grayscale images, 10 classes) |
| Model | Simple CNN (~21K parameters) |
| Algorithm | FedAvg (Federated Averaging) |
| Rounds | 5 |
| Epochs per round | 1 |
| Expected accuracy | ~95%+ |
| Runtime | ~2-3 minutes |

**Submit:**
```bash
nvflare job submit -j jobs/configs/pt_mnist_fedavg
```

### CIFAR-10 ResNet-18 (`configs/pt_cifar10_resnet18/`)

A GPU-intensive job for stress-testing the cluster.

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 (32x32 color images, 10 classes) |
| Model | ResNet-18 (~11M parameters) |
| Algorithm | FedAvg (Federated Averaging) |
| Rounds | 10 |
| Epochs per round | 10 |
| Expected accuracy | ~50-60% |
| Runtime | ~10-15 minutes |

**Submit:**
```bash
nvflare job submit -j jobs/configs/pt_cifar10_resnet18
```

---

## Generating Custom Jobs

Use the generator scripts to create job configurations tailored to your `project.yml`:

### Generate MNIST Job

```bash
cd jobs/generators
python fedavg_mnist.py -p ../../project.yml
```

This creates `job_config/pt_mnist_fedavg/` with configurations for all clients defined in `project.yml`.

### Generate CIFAR-10 Job

```bash
cd jobs/generators
python fedavg_cifar10.py -p ../../project.yml
```

### Submit Generated Job

```bash
nvflare job submit -j jobs/generators/job_config/pt_mnist_fedavg
```

---

## Job Configuration Structure

Each job in `configs/` follows this structure:

```
pt_mnist_fedavg/
├── meta.json                   # Job metadata and deployment map
└── app/
    ├── config/
    │   ├── config_fed_server.json  # Server workflow configuration
    │   └── config_fed_client.json  # Client executor configuration
    └── custom/
        └── src/
            ├── pt_mnist_fl.py      # Training script
            └── mnist_cnn.py        # Model definition
```

### meta.json

Defines job metadata and which participants receive which app:

```json
{
    "name": "pt_mnist_fedavg",
    "resource_spec": {},
    "min_clients": 1,
    "deploy_map": {
        "app": ["@ALL"]
    }
}
```

- `"@ALL"` deploys the app to both server and all clients
- For separate server/client apps, use specific participant names

### config_fed_server.json

Configures the server-side workflow (FedAvg aggregation):

```json
{
    "format_version": 2,
    "workflows": [
        {
            "id": "scatter_and_gather",
            "path": "nvflare.app_common.workflows.fedavg.FedAvg",
            "args": {
                "num_clients": 6,
                "num_rounds": 5
            }
        }
    ],
    "components": [
        {
            "id": "persistor",
            "path": "nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor",
            "args": {
                "model": {
                    "path": "src.mnist_cnn.MnistCnn"
                }
            }
        }
    ]
}
```

### config_fed_client.json

Configures the client-side executor (training):

```json
{
    "format_version": 2,
    "executors": [
        {
            "tasks": ["*"],
            "executor": {
                "path": "nvflare.app_opt.pt.client_api_launcher_executor.PTClientAPILauncherExecutor",
                "args": {
                    "launcher_id": "launcher"
                }
            }
        }
    ],
    "components": [
        {
            "id": "launcher",
            "path": "nvflare.app_common.launchers.subprocess_launcher.SubprocessLauncher",
            "args": {
                "script": "custom/src/pt_mnist_fl.py"
            }
        }
    ]
}
```

---

## Training Scripts

Training scripts in `src/` use the NVFlare Client API:

### Basic Pattern

```python
import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import ParamsType

# Initialize NVFlare client
flare.init()

# Get client info
sys_info = flare.system_info()
client_name = sys_info["site_name"]

# Load client-specific data
train_data = torch.load(f"/data/mnist/{client_name}_train.pt")

# Training loop
while flare.is_running():
    # Receive global model from server
    input_model = flare.receive()
    round_num = input_model.current_round

    # Load weights into local model
    model.load_state_dict(input_model.params)

    # Train on local data
    for epoch in range(epochs_per_round):
        for batch in train_loader:
            # ... training code ...

    # Evaluate
    accuracy = evaluate(model, test_loader)

    # Send updated model back to server
    output_model = flare.FLModel(
        params=model.cpu().state_dict(),
        params_type=ParamsType.FULL,  # Required!
        metrics={"accuracy": accuracy}
    )
    flare.send(output_model)
```

### Key Points

1. **Data Loading**: Scripts expect data at `/data/{dataset}/{client_name}_train.pt`
   - This is mounted from `~/nvflare_data/{dataset}/` on the host
   - Use `scripts/distribute_splits.py` to distribute data

2. **ParamsType.FULL**: Always set `params_type=ParamsType.FULL` when sending model updates

3. **Client Name**: Use `flare.system_info()["site_name"]` to get the client's name for loading client-specific data

4. **GPU Usage**: Scripts auto-detect CUDA availability with `torch.cuda.is_available()`

---

## Model Definitions

### MNIST CNN (`src/mnist_cnn.py`)

Simple 2-layer CNN for MNIST:
- ~21K parameters
- Input: 28x28 grayscale
- Output: 10 classes

### ResNet-18 for CIFAR-10 (`src/resnet18_cifar10.py`)

ResNet-18 adapted for smaller CIFAR-10 images:
- ~11M parameters
- Modified first conv layer (3x3 instead of 7x7)
- Removed initial max pooling
- Input: 32x32 RGB
- Output: 10 classes

**Important**: The model class copies layers as direct attributes (not wrapped in `self.model`) to ensure state_dict keys match between server and clients.

---

## Data Distribution

Data must be distributed before running jobs. Use the distribution script:

```bash
# View help
python scripts/distribute_splits.py --help

# MNIST with uniform split (IID)
python scripts/distribute_splits.py --dataset mnist --split_method uniform

# CIFAR-10 with square split (non-IID)
python scripts/distribute_splits.py --dataset cifar10 --split_method square
```

### Split Methods

| Method | Description | Data Distribution |
|--------|-------------|-------------------|
| `uniform` | Equal data per client | IID (Independent and Identically Distributed) |
| `linear` | Linearly increasing amounts | Mild heterogeneity |
| `square` | Quadratically increasing | Moderate heterogeneity |
| `exponential` | Exponentially increasing | Extreme heterogeneity |

### Verify Data Distribution

```bash
# Check data on all nodes
ansible -i ansible/inventory.ini all -a "ls -lh ~/nvflare_data/mnist/"
```

---

## Troubleshooting

### "Training data not found"

```
FileNotFoundError: Training data not found: /data/mnist/k3s-client-09_train.pt
```

**Solution**: Run data distribution first:
```bash
python scripts/distribute_splits.py --dataset mnist --split_method uniform
```

### "Missing key 'conv1.weight'"

Model state_dict keys don't match between server and client.

**Solution**: Ensure the model class in the job config matches the one used in training scripts. For wrapped models (like ResNet), layers must be copied as direct attributes.

### Job Stuck or No Progress

Check container logs:
```bash
docker logs nvflare-server --tail 100
docker logs nvflare-client-09 --tail 100
```

Check if all clients are connected:
```bash
python scripts/job_status.py <job_id>
```

---

## Creating New Jobs

1. **Create model definition** in `src/`
2. **Create training script** in `src/` using NVFlare Client API
3. **Create job directory** in `configs/` with:
   - `meta.json`
   - `app/config/config_fed_server.json`
   - `app/config/config_fed_client.json`
   - `app/custom/src/` (copy training script and model)
4. **Distribute data** for your dataset
5. **Submit**: `nvflare job submit -j jobs/configs/your_job`

Or use a generator script as a template - modify `generators/fedavg_mnist.py` for your use case.
