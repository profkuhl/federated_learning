# Examples: 

## TorchVision MNIST

```python
import torchvision
import torchvision.transforms as transforms
import data_utils # Your new general script

print("Loading MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Use torchvision to download
_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# --- This is the key "extraction" step ---
# We apply the transform to ALL data at once to create the final tensors
print("Applying transforms to create final tensors...")
train_loader = torch.utils.data.DataLoader(_train, batch_size=len(_train))
train_data, train_labels = next(iter(train_loader))

test_loader = torch.utils.data.DataLoader(_test, batch_size=len(_test))
test_data, test_labels = next(iter(test_loader))

print(f"Train tensors: {train_data.shape}, {train_labels.shape}")
print(f"Test tensors: {test_data.shape}, {test_labels.shape}")
# --- End of extraction ---

# Call the general utility
data_utils.split_and_distribute(
    train_data=train_data,
    train_labels=train_labels,
    test_data=test_data,
    test_labels=test_labels,
    inventory_path="inventory.ini",
    split_method="uniform",
    remote_dest_path="/tmp/mnist_data"
)
```

# Example SKLearn

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import data_utils # Your new general script

print("Loading Sklearn digits dataset...")

# --- This is the key "extraction" step ---
X, y = load_digits(return_X_y=True)

# Sklearn gives one big dataset, so we split it into train/test first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train arrays: {X_train.shape}, {y_train.shape}")
print(f"Test arrays: {X_test.shape}, {y_test.shape}")
# --- End of extraction ---

# Call the general utility
# It's fine to pass NumPy arrays; the script will convert them
data_utils.split_and_distribute(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    inventory_path="inventory.ini",
    split_method="exponential",
    remote_dest_path="/tmp/sklearn_digits_data"
)
```

# Example Client-Side

```python
import torch
import nvflare.client as flare
from torch.utils.data import DataLoader, TensorDataset

# --- Inside your client's training logic ---

flare.init()
sys_info = flare.system_info()
client_name = sys_info["site_name"] # e.g., "site-1"

# This path must match what you provided to split_and_distribute()
DATA_PATH = "/tmp/mnist_data" # Or "/tmp/sklearn_digits_data", etc.

# 1. Load this client's specific training data
train_data, train_labels = torch.load(f"{DATA_PATH}/{client_name}_train.pt")
client_train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(client_train_dataset, batch_size=64, shuffle=True)

# 2. Load the shared test data
test_data, test_labels = torch.load(f"{DATA_PATH}/test_data.pt")
client_test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(client_test_dataset, batch_size=64, shuffle=False)

print(f"Client {client_name} loaded {len(client_train_dataset)} training samples.")

# ... rest of your training loop ...
```
