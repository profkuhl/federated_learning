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

"""ResNet-18 model adapted for CIFAR-10 (32x32 images)."""

from torch import nn
from torchvision.models import resnet18


def create_resnet18_cifar10():
    """Create ResNet-18 adapted for CIFAR-10's 32x32 images."""
    model = resnet18(weights=None, num_classes=10)
    # Adjust first conv layer for smaller CIFAR-10 images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for small images
    return model


# NVFlare expects a class that can be instantiated
class ResNet18Cifar10(nn.Module):
    """ResNet-18 modified for CIFAR-10's 32x32 images.

    Note: This class directly copies the parameters from a standard ResNet-18
    to ensure state_dict keys match between server and clients.
    """

    def __init__(self):
        super().__init__()
        # Get a base ResNet-18
        base_model = resnet18(weights=None, num_classes=10)

        # Copy all layers as direct attributes (not nested in self.model)
        # This ensures state_dict keys don't have a 'model.' prefix
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = nn.Identity()  # Remove maxpool for small images
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        self.fc = base_model.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
