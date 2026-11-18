"""Train the diffusion model on CIFAR-10 and save it for the API.

Run from the project root:
    uv run python train_diffusion.py
"""
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from app.diffusion_model import UNet, DiffusionModel

EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHTS_PATH = Path("app") / "diffusion_weights.pth"
SAMPLES_DIR = Path("diffusion_samples")
SAMPLES_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

# CIFAR-10 images are 32x32 RGB; pixel values in [0, 1]
transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Normalization stats so the training images have mean 0 and variance 1
all_pixels = torch.stack([train_dataset[i][0] for i in range(5000)])
mean, std = all_pixels.mean().item(), all_pixels.std().item()
print(f"Normalization - mean: {mean:.4f}, std: {std:.4f}")

model = DiffusionModel(UNet()).to(device)
model.normalizer_mean, model.normalizer_std = mean, std

optimizer = torch.optim.AdamW(model.network.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
loss_fn = nn.L1Loss()

for epoch in range(EPOCHS):
    losses = [model.train_step(images.to(device), optimizer, loss_fn) for images, _ in train_loader]
    print(f"Epoch {epoch + 1}/{EPOCHS} - train loss: {sum(losses) / len(losses):.4f}")

    samples = model.generate(num_images=64, diffusion_steps=20)
    save_image(samples, SAMPLES_DIR / f"epoch_{epoch + 1:02d}.png", nrow=8)

# Save only what the API needs: the EMA network and the normalization stats
torch.save({
    "ema_model_state_dict": model.ema_network.to("cpu").state_dict(),
    "normalizer_mean": mean,
    "normalizer_std": std,
}, WEIGHTS_PATH)
print(f"Saved diffusion weights to {WEIGHTS_PATH}")