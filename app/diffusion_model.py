"""Diffusion model (Module 8, Practical 2), adapted to 32x32 CIFAR-10 color images.

Forward process: add Gaussian noise to an image following a diffusion schedule.
A U-Net learns to predict the noise that was added. To generate, we start from
pure noise and remove the predicted noise step by step (reverse diffusion).
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 32
NUM_CHANNELS = 3  # RGB


# ---------------------------------------------------------------------------
# Diffusion schedule: how much signal / noise there is at time t in [0, 1]
# ---------------------------------------------------------------------------
def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    original_shape = diffusion_times.shape
    t = diffusion_times.flatten()
    start_angle = torch.acos(torch.tensor(max_signal_rate, dtype=torch.float32, device=t.device))
    end_angle = torch.acos(torch.tensor(min_signal_rate, dtype=torch.float32, device=t.device))
    angles = start_angle + t * (end_angle - start_angle)
    signal_rates = torch.cos(angles).reshape(original_shape)
    noise_rates = torch.sin(angles).reshape(original_shape)
    return noise_rates, signal_rates


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------
class SinusoidalEmbedding(nn.Module):
    """Encodes the noise variance as a vector of sines and cosines."""

    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):  # x: (B, 1, 1, 1) -> (B, 1, 1, 2 * num_frequencies)
        x = x.expand(-1, 1, 1, self.num_frequencies)
        return torch.cat([torch.sin(self.angular_speeds * x), torch.cos(self.angular_speeds * x)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.proj(x)
        x = self.conv1(x)
        x = x * torch.sigmoid(x)  # swish
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return self.pool(x)


class UpBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width

    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        for block in self.blocks:
            x = torch.cat([x, skips.pop()], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    """Predicts the noise in a noisy image, given the noise variance."""

    def __init__(self, image_size=IMAGE_SIZE, num_channels=NUM_CHANNELS):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.embedding = SinusoidalEmbedding(num_frequencies=16)  # -> 32 channels

        self.down1 = DownBlock(32, block_depth=2, in_channels=64)  # 32 image + 32 embedding channels
        self.down2 = DownBlock(64, block_depth=2, in_channels=32)
        self.down3 = DownBlock(96, block_depth=2, in_channels=64)

        self.mid1 = ResidualBlock(96, 128)
        self.mid2 = ResidualBlock(128, 128)

        self.up1 = UpBlock(96, block_depth=2, in_channels=128)
        self.up2 = UpBlock(64, block_depth=2, in_channels=96)
        self.up3 = UpBlock(32, block_depth=2, in_channels=64)

        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        noise_emb = self.embedding(noise_variances)  # (B, 1, 1, 32)
        noise_emb = F.interpolate(noise_emb.permute(0, 3, 1, 2), size=(self.image_size, self.image_size), mode="nearest")
        x = torch.cat([x, noise_emb], dim=1)

        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)
        x = self.mid1(x)
        x = self.mid2(x)
        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)
        return self.final(x)


# ---------------------------------------------------------------------------
# Diffusion model: training step and generation
# ---------------------------------------------------------------------------
class DiffusionModel(nn.Module):
    def __init__(self, network, schedule_fn=offset_cosine_diffusion_schedule, ema_decay=0.99):
        super().__init__()
        self.network = network
        self.ema_network = copy.deepcopy(network)  # smoothed copy used for generation
        self.ema_network.eval()
        self.ema_decay = ema_decay
        self.schedule_fn = schedule_fn
        self.normalizer_mean = 0.0
        self.normalizer_std = 1.0

    def denormalize(self, x):
        return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        network = self.network if training else self.ema_network
        network.train(training)
        pred_noises = network(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        pred_images = initial_noise
        for step in range(diffusion_steps):
            t = torch.ones((initial_noise.shape[0], 1, 1, 1), device=initial_noise.device) * (1 - step * step_size)
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)
            next_noise_rates, next_signal_rates = self.schedule_fn(t - step_size)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return pred_images

    @torch.no_grad()
    def generate(self, num_images, diffusion_steps=20):
        device = next(self.parameters()).device
        noise = torch.randn((num_images, self.network.num_channels, self.network.image_size, self.network.image_size), device=device)
        return self.denormalize(self.reverse_diffusion(noise, diffusion_steps))

    def train_step(self, images, optimizer, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)
        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
        loss = loss_fn(pred_noises, noises)  # learn to predict the noise that was added

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the smoothed (EMA) copy of the weights
        with torch.no_grad():
            for ema_p, p in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_p.copy_(self.ema_decay * ema_p + (1.0 - self.ema_decay) * p)
        return loss.item()