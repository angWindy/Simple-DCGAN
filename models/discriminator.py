import torch
import torch.nn as nn

from config import CONFIG


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input is image (nc) x 64 x 64
            nn.Conv2d(CONFIG["nc"], CONFIG["ndf"], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf) x 32 x 32
            nn.Conv2d(CONFIG["ndf"], CONFIG["ndf"] * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(CONFIG["ndf"] * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*2) x 16 x 16
            nn.Conv2d(CONFIG["ndf"] * 2, CONFIG["ndf"] * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(CONFIG["ndf"] * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*4) x 8 x 8
            nn.Conv2d(CONFIG["ndf"] * 4, CONFIG["ndf"] * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(CONFIG["ndf"] * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*8) x 4 x 4
            nn.Conv2d(CONFIG["ndf"] * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)
