"""Train the Assignment 3 GAN on MNIST and save the generator for the API.

Run from the project root:
    uv run python train_gan.py
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from app.gan_model import Generator, Discriminator, LATENT_DIM

EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
WEIGHTS_PATH = Path("app") / "gan_generator.pth"
SAMPLES_DIR = Path("gan_samples")
SAMPLES_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)

# Use Apple GPU (mps), NVIDIA GPU (cuda) or CPU, whichever is available
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

# MNIST: 60,000 training images of hand-written digits, 28x28 grayscale.
# Normalize to [-1, 1] to match the generator's Tanh output.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
opt_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# Fixed noise so we can watch the same samples improve across epochs
fixed_noise = torch.randn(64, LATENT_DIM, device=device)

for epoch in range(EPOCHS):
    generator.train()
    discriminator.train()
    d_loss_total, g_loss_total = 0.0, 0.0

    for real_images, _ in train_loader:  # labels are not used: GANs are unsupervised
        real_images = real_images.to(device)
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # ---- Train the discriminator: real images -> 1, generated images -> 0 ----
        noise = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_images = generator(noise)

        opt_d.zero_grad()
        d_loss_real = criterion(discriminator(real_images), real_labels)
        d_loss_fake = criterion(discriminator(fake_images.detach()), fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        opt_d.step()

        # ---- Train the generator: make the discriminator say "real" (1) ----
        opt_g.zero_grad()
        g_loss = criterion(discriminator(fake_images), real_labels)
        g_loss.backward()
        opt_g.step()

        d_loss_total += d_loss.item()
        g_loss_total += g_loss.item()

    n = len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} - D loss: {d_loss_total / n:.4f} | G loss: {g_loss_total / n:.4f}")

    # Save a grid of samples to see progress
    generator.eval()
    with torch.no_grad():
        samples = generator(fixed_noise)
    save_image(samples, SAMPLES_DIR / f"epoch_{epoch + 1:02d}.png", nrow=8, normalize=True, value_range=(-1, 1))

# Save only the generator: it is all the API needs to create new digits
torch.save(generator.to("cpu").state_dict(), WEIGHTS_PATH)
print(f"Saved generator weights to {WEIGHTS_PATH}")
print(f"Sample images saved in {SAMPLES_DIR}/")