import torch
import torch.nn as nn

from config import CONFIG


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is z vector, pass through first transposed convolution
            nn.ConvTranspose2d(CONFIG["nz"], CONFIG["ngf"] * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(CONFIG["ngf"] * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(
                CONFIG["ngf"] * 8, CONFIG["ngf"] * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(CONFIG["ngf"] * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(
                CONFIG["ngf"] * 4, CONFIG["ngf"] * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(CONFIG["ngf"] * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(CONFIG["ngf"] * 2, CONFIG["ngf"], 4, 2, 1, bias=False),
            nn.BatchNorm2d(CONFIG["ngf"]),
            nn.ReLU(True),
            # State size: (ngf) x 32 x 32
            nn.ConvTranspose2d(CONFIG["ngf"], CONFIG["nc"], 4, 2, 1, bias=False),
            nn.Tanh(),
            # State size: (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
