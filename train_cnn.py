"""Train the Assignment 2 CNN on CIFAR-10 and save the weights for the API.

Run from the project root:
    uv run python train_cnn.py
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from app.cnn_model import CNN, TRANSFORM

EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHTS_PATH = Path("app") / "cnn_weights.pth"

torch.manual_seed(42)

# Use Apple GPU (mps), NVIDIA GPU (cuda) or CPU, whichever is available
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

# CIFAR-10: 50,000 training and 10,000 test images, 10 classes
train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=TRANSFORM)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=TRANSFORM)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}/{EPOCHS} - train loss: {running_loss / len(train_loader):.4f}")

# Evaluate on the test set
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
print(f"Test accuracy: {100 * correct / total:.2f}%")

# Save weights (moved to CPU so the API can load them anywhere)
torch.save(model.to("cpu").state_dict(), WEIGHTS_PATH)
print(f"Saved model weights to {WEIGHTS_PATH}")