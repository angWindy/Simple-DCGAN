import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

# Set seed to make results reproducible
random.seed(42)
torch.manual_seed(42)

# Configuration
CONFIG = {
    # Data
    "workers": 2,
    "batch_size": 128,
    "img_size": 64,
    "nc": 3,  # Number of channels
    "nz": 100,  # Size of z latent vector
    "ngf": 64,  # Size of feature maps in generator
    "ndf": 64,  # Size of feature maps in discriminator
    # Training
    "n_critic": 5,
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,  # Weight decay for AdamW Optimizers
    "beta1": 0.5,  # Beta1 for Adam optimizer
}

dataroot = (
    "/kaggle/input/celebahq-resized-256x256/celeba_hq_256"  # Directory with dataset
)

# Create directory to save images
os.makedirs("/kaggle/working/images", exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Định nghĩa các biến đổi
transforms_list = [
    transforms.Resize(CONFIG["img_size"]),
    transforms.CenterCrop(CONFIG["img_size"]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize from -1 to 1
]


# Define Dataset Class
class CelebaDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(root_dir) if f.endswith((".jpg", ".png", ".jpeg"))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image, 0  # Return 0 as dummy label


# Create dataset
dataset = CelebaDataset(
    root_dir=dataroot, transform=transforms.Compose(transforms_list)
)

# Create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=CONFIG["workers"],
)


# Function to initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator
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


# Discriminator
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


# Initialize models
netG = Generator().to(device)
netG.apply(weights_init)

netD = Discriminator().to(device)
netD.apply(weights_init)

# Định nghĩa loss function và optimizers
criterion = nn.BCELoss()
fixed_noise = torch.randn(64, CONFIG["nz"], 1, 1, device=device)
real_label = 1
fake_label = 0

# AdamW Optimizer
optimizerD = optim.AdamW(
    netD.parameters(),
    lr=CONFIG["learning_rate"],
    betas=(CONFIG["beta1"], 0.999),
    weight_decay=CONFIG["weight_decay"],
)
optimizerG = optim.AdamW(
    netG.parameters(),
    lr=CONFIG["learning_rate"],
    betas=(CONFIG["beta1"], 0.999),
    weight_decay=CONFIG["weight_decay"],
)

# Save training process
img_list = []
G_losses = []
D_losses = []
iters = 0

# Training loop
print("Training...")
for epoch in range(CONFIG["num_epochs"]):
    # Thêm progress bar cho mỗi epoch
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
    for i, data in enumerate(pbar, 0):
        ############################
        # (1) Update D: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with real images
        netD.zero_grad()
        real = data[0].to(device)
        batch_size = real.size(0)
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)

        output = netD(real).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with fake images
        noise = torch.randn(batch_size, CONFIG["nz"], 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)

        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G: maximize log(D(G(z)))
        ###########################
        if i % CONFIG["n_critic"] == 0:
            netG.zero_grad()
            label.fill_(real_label)  # fake label for generator

            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()

        # Cập nhật progress bar
        pbar.set_postfix(
            {
                "Loss_D": f"{errD.item():.4f}",
                "Loss_G": f"{errG.item():.4f}",
                "D(x)": f"{D_x:.4f}",
                "D(G(z))": f"{D_G_z1:.4f}/{D_G_z2:.4f}",
            }
        )

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        iters += 1

    # After each epoch, save and display generated images
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    # Save images each epoch
    vutils.save_image(
        fake, f"/kaggle/working/images/epoch_{epoch+1:03d}.png", normalize=True
    )

# Save models
torch.save(netG.state_dict(), "/kaggle/working/generator.pth")
torch.save(netD.state_dict(), "/kaggle/working/discriminator.pth")

# Display loss
plt.figure(figsize=(10, 5))
plt.title("Generator và Discriminator Loss")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("/kaggle/working/loss_plot.png")
plt.show()

# Display progress of images - chỉ hiển thị 5 ảnh
plt.figure(figsize=(15, 3))
total_images = len(img_list)
if total_images <= 5:
    selected_indices = list(range(total_images))
else:
    # Chọn 5 ảnh đều nhau: 1, 5, 10, 15, 20 (hoặc tương tự)
    step = (total_images - 1) // 4
    selected_indices = [0] + [i * step for i in range(1, 5)]

for i, idx in enumerate(selected_indices):
    plt.subplot(1, 5, i + 1)
    plt.axis("off")
    plt.title(f"Epoch {idx+1}")
    plt.imshow(np.transpose(img_list[idx], (1, 2, 0)))
plt.savefig("/kaggle/working/progress.png")
plt.show()

# Display final result
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Generated images by Generator")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.savefig("/kaggle/working/final_result.png")
plt.show()
