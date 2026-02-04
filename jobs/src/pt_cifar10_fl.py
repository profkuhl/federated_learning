# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GPU-intensive CIFAR-10 training with ResNet-18 for federated learning.
Designed to stress-test GPU clusters with ~10 minute runtime.
"""

import os
import time

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import ParamsType
from nvflare.client.tracking import SummaryWriter

# Use ResNet-18 from torchvision for GPU-intensive training
from torchvision.models import resnet18

# Standard data path - mounted from ~/nvflare_data on host to /data in container
DATASET_PATH = os.environ.get("NVFLARE_DATA_PATH", "/data/cifar10")


def main():
    # Training hyperparameters - tuned for ~10 min total runtime with 6 clients
    batch_size = 64
    epochs_per_round = 10  # More epochs = more GPU work
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4

    # Initialize model - ResNet-18 has ~11M parameters (vs ~20K for simple CNN)
    # Modified for CIFAR-10's 32x32 images
    model = resnet18(weights=None, num_classes=10)
    # Adjust first conv layer for smaller CIFAR-10 images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for small images

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]

    # Load this client's specific training data
    train_file = Path(DATASET_PATH) / f"{client_name}_train.pt"
    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")

    train_data, train_labels = torch.load(train_file)
    client_train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(
        client_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Load the shared test data
    test_file = Path(DATASET_PATH) / "test_data.pt"
    if not test_file.exists():
        raise FileNotFoundError(f"Test data not found: {test_file}")

    test_data, test_labels = torch.load(test_file)
    client_test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(
        client_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"\n{'='*60}")
    print(f"Client: {client_name}")
    print(f"Training samples: {len(client_train_dataset):,}")
    print(f"Test samples: {len(client_test_dataset):,}")
    print(f"Epochs per round: {epochs_per_round}")
    print(f"Batch size: {batch_size}")
    print(f"Model: ResNet-18 (~11M parameters)")
    print(f"{'='*60}\n")

    summary_writer = SummaryWriter()
    round_num = 0

    while flare.is_running():
        input_model = flare.receive()
        round_num = input_model.current_round
        print(f"\n[Round {round_num}] Starting training...")
        round_start = time.time()

        # Load global model weights
        model.load_state_dict(input_model.params)
        model.to(device)
        model.train()

        # Learning rate scheduler - cosine annealing within each round
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs_per_round * len(train_loader))

        total_samples = 0
        total_loss = 0.0

        for epoch in range(epochs_per_round):
            epoch_loss = 0.0
            epoch_samples = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                batch_loss = loss.item() * images.size(0)
                epoch_loss += batch_loss
                epoch_samples += images.size(0)

            avg_epoch_loss = epoch_loss / epoch_samples
            total_loss += epoch_loss
            total_samples += epoch_samples

            # Log every epoch
            global_step = round_num * epochs_per_round + epoch
            summary_writer.add_scalar("train/loss", avg_epoch_loss, global_step)
            summary_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            print(f"  Epoch {epoch+1}/{epochs_per_round} - Loss: {avg_epoch_loss:.4f}")

        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        test_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_test_loss = test_loss / total

        summary_writer.add_scalar("val/accuracy", accuracy, round_num)
        summary_writer.add_scalar("val/loss", avg_test_loss, round_num)

        round_time = time.time() - round_start
        print(f"  Validation - Accuracy: {accuracy:.2f}%, Loss: {avg_test_loss:.4f}")
        print(f"  Round completed in {round_time:.1f}s")

        # Send updated model back to server
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            params_type=ParamsType.FULL,
            metrics={"accuracy": accuracy / 100.0},  # Normalize to 0-1 range
            meta={"NUM_STEPS_CURRENT_ROUND": epochs_per_round * len(train_loader)},
        )

        flare.send(output_model)

    print(f"\nTraining complete!")


if __name__ == "__main__":
    main()
