import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# CIFAR-10 class names (index -> label)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Shared preprocessing for training and for the API.
# CIFAR-10 images are 32x32, but the required architecture expects 64x64x3,
# so we resize every image to 64x64 before feeding it to the network.
TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class CNN(nn.Module):
    """CNN that follows the Assignment 2 architecture specification.

    Input: RGB image of size 64x64x3
    Conv2D(16 filters, 3x3, stride 1, padding 1) -> ReLU -> MaxPool(2x2, stride 2)
    Conv2D(32 filters, 3x3, stride 1, padding 1) -> ReLU -> MaxPool(2x2, stride 2)
    Flatten -> FC(100) -> ReLU -> FC(10)
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (3, 64, 64) -> (16, 64, 64) -> (16, 32, 32)
        x = self.pool(F.relu(self.conv2(x)))  # (16, 32, 32) -> (32, 32, 32) -> (32, 16, 16)
        x = self.flatten(x)                   # (32, 16, 16) -> 8192
        x = F.relu(self.fc1(x))               # 8192 -> 100
        x = self.fc2(x)                       # 100 -> 10 (class scores / logits)
        return x