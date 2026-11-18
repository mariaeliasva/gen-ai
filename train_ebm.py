"""Train the Energy-Based Model on CIFAR-10 and save it for the API.

Run from the project root:
    uv run python train_ebm.py
"""
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from app.ebm_model import EnergyModel, generate_samples, IMAGE_SIZE, NUM_CHANNELS

EPOCHS = 20           
BATCH_SIZE = 128
ALPHA = 0.1          # regularization weight
STEPS = 60           # Langevin steps per training batch
STEP_SIZE = 10
NOISE = 0.005
WEIGHTS_PATH = Path("app") / "ebm_weights.pth"
SAMPLES_DIR = Path("ebm_samples")
SAMPLES_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

# CIFAR-10 images are already 32x32; scale pixels to [-1, 1]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


def random_images(n):
    return torch.rand((n, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), device=device) * 2 - 1


class Buffer:
    """Keeps previously generated samples so Langevin dynamics doesn't restart from noise every time."""

    def __init__(self, model):
        self.model = model
        self.examples = [random_images(1) for _ in range(BATCH_SIZE)]

    def sample_new_examples(self):
        n_new = np.random.binomial(BATCH_SIZE, 0.05)  # ~5% fresh random images
        old_imgs = torch.cat(random.choices(self.examples, k=BATCH_SIZE - n_new), dim=0)
        inp_imgs = torch.cat([random_images(n_new), old_imgs], dim=0)
        new_imgs = generate_samples(self.model, inp_imgs, STEPS, STEP_SIZE, NOISE)
        self.examples = (list(torch.split(new_imgs, 1, dim=0)) + self.examples)[:8192]
        return new_imgs


model = EnergyModel().to(device)
buffer = Buffer(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))

for epoch in range(EPOCHS):
    model.train()
    totals = {"loss": 0.0, "real": 0.0, "fake": 0.0}
    for real_imgs, _ in train_loader:
        real_imgs = (real_imgs.to(device) + torch.randn_like(real_imgs.to(device)) * NOISE).clamp(-1.0, 1.0)
        fake_imgs = buffer.sample_new_examples()

        out = model(torch.cat([real_imgs, fake_imgs], dim=0))
        real_out, fake_out = torch.split(out, [real_imgs.size(0), fake_imgs.size(0)], dim=0)

        # Contrastive divergence: push energy DOWN on real images, UP on generated ones
        cdiv_loss = real_out.mean() - fake_out.mean()
        reg_loss = ALPHA * (real_out.pow(2).mean() + fake_out.pow(2).mean())
        loss = cdiv_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        totals["loss"] += loss.item()
        totals["real"] += real_out.mean().item()
        totals["fake"] += fake_out.mean().item()

    n = len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {totals['loss'] / n:.4f} | "
          f"real energy: {totals['real'] / n:.4f} | fake energy: {totals['fake'] / n:.4f}")

    # Generate fresh samples from noise to see progress
    samples = generate_samples(model, random_images(64), steps=256, step_size=10.0, noise_std=0.005)
    save_image(samples, SAMPLES_DIR / f"epoch_{epoch + 1:02d}.png", nrow=8, normalize=True, value_range=(-1, 1))

torch.save(model.to("cpu").state_dict(), WEIGHTS_PATH)
print(f"Saved EBM weights to {WEIGHTS_PATH}")