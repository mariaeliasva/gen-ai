"""Energy-Based Model (Module 8, Practical 1) for 32x32 CIFAR-10 color images.

The network maps an image x to a single number E(x): its "energy".
Training teaches it to give LOW energy to real images and HIGH energy to fake ones.
To generate, we start from random noise and run gradient descent on the IMAGE
(not on the weights) to lower its energy: Langevin dynamics.
"""
import torch
import torch.nn as nn

IMAGE_SIZE = 32
NUM_CHANNELS = 3  # RGB


def swish(x):
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    """CNN that maps a (3, 32, 32) image to a scalar energy."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(NUM_CHANNELS, 16, kernel_size=5, stride=2, padding=2)   # 32 -> 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 16 -> 8
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 8 -> 4
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # 4 -> 2
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)


def generate_samples(energy_model, inp_imgs, steps, step_size, noise_std):
    """Langevin dynamics: move the input images toward low energy.

    Key idea of the assignment: here the gradient is taken with respect to the
    INPUT IMAGES, not the network weights. The weights stay fixed.
    """
    energy_model.eval()

    # Freeze the weights: we don't want gradients for them during sampling
    for w in energy_model.parameters():
        w.requires_grad_(False)

    inp_imgs = inp_imgs.detach()
    for _ in range(steps):
        # 1. Add a little noise (no gradient tracking needed for this step)
        with torch.no_grad():
            inp_imgs = (inp_imgs + torch.randn_like(inp_imgs) * noise_std).clamp(-1.0, 1.0)

        # 2. Tell PyTorch to track gradients for the IMAGES
        inp_imgs.requires_grad_(True)

        # 3. Energy of each image, and its gradient with respect to the pixels
        energy = energy_model(inp_imgs)
        grads, = torch.autograd.grad(energy, inp_imgs, grad_outputs=torch.ones_like(energy))

        # 4. Gradient DEscent on the pixels (clip the gradient for stability)
        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)
            inp_imgs = (inp_imgs - step_size * grads).clamp(-1.0, 1.0)

    # Unfreeze the weights so training can continue
    for w in energy_model.parameters():
        w.requires_grad_(True)

    return inp_imgs.detach()