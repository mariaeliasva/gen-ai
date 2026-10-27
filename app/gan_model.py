import torch.nn as nn

# Size of the random noise vector the generator receives
LATENT_DIM = 100


class Generator(nn.Module):
    """GAN generator that follows the Assignment 3 architecture.

    Input: noise vector of shape (batch_size, 100)
    Linear -> 7x7x128, reshape
    ConvTranspose2d 128 -> 64 (k=4, s=2, p=1) -> 14x14, BatchNorm2d, ReLU
    ConvTranspose2d 64 -> 1   (k=4, s=2, p=1) -> 28x28, Tanh
    """

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),    # 14x14 -> 28x28
            nn.Tanh(),  # pixel values in [-1, 1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)  # reshape to (batch, 128, 7, 7)
        return self.net(x)


class Discriminator(nn.Module):
    """GAN discriminator that follows the Assignment 3 architecture.

    Input: image of shape (1, 28, 28)
    Conv2d 1 -> 64   (k=4, s=2, p=1) -> 14x14, LeakyReLU(0.2)
    Conv2d 64 -> 128 (k=4, s=2, p=1) -> 7x7, BatchNorm2d, LeakyReLU(0.2)
    Flatten -> Linear -> single output (probability that the image is real)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),    # 28x28 -> 14x14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),  # probability that the image is real
        )

    def forward(self, x):
        return self.net(x)