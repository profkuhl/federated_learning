import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import data_utils


train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

print("Applying transforms to create final tensors...")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset))
train_data, train_labels = next(iter(train_loader))

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset))
test_data, test_labels = next(iter(test_loader))

print(f"Train tensors: {train_data.shape}, {train_labels.shape}")
print(f"Test tensors: {test_data.shape}, {test_labels.shape}")

data_utils.split_and_distribute(
    train_data=train_data,
    train_labels=train_labels,
    test_data=test_data,
    test_labels=test_labels,
    inventory_path="/home/k3s-server-07/federated_learning/ansible/inventory.ini",
    split_method="square",
    remote_dest_path="/tmp/mnist_data"
)
