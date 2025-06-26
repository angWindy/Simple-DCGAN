import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm

# Thêm đường dẫn gốc vào sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from data_utils import CelebaDataset, get_transforms
from models import Discriminator, Generator
from utils import weights_init

# Set seed for reproducibility
random.seed(42)
torch.manual_seed(42)


def main():
    # Kiểm tra xem dataset đã được tải chưa
    if not os.path.exists(CONFIG["celeba_dir"]):
        print(f"Dataset not found at {CONFIG['celeba_dir']}")
        print("Please run 'python download_data.py' first to download the dataset.")
        return

    # Create output directories
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["sample_dir"], exist_ok=True)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = CelebaDataset(transform=get_transforms())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["workers"],
    )

    # Initialize models
    netG = Generator().to(device)
    netG.apply(weights_init)
    print(netG)

    netD = Discriminator().to(device)
    netD.apply(weights_init)
    print(netD)

    # Define loss function and optimizers
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, CONFIG["nz"], 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # AdamW Optimizers
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

    # Training variables
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(CONFIG["num_epochs"]):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        for i, data in enumerate(pbar, 0):
            ############################
            # (1) Update D: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with real images
            netD.zero_grad()
            real = data[0].to(device)
            batch_size = real.size(0)
            label = torch.full(
                (batch_size,), real_label, dtype=torch.float, device=device
            )

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
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost

            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()

            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss_D": f"{errD.item():.4f}",
                    "Loss_G": f"{errG.item():.4f}",
                    "D(x)": f"{D_x:.4f}",
                    "D(G(z))": f"{D_G_z1:.4f}/{D_G_z2:.4f}",
                }
            )

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1

        # After each epoch, generate and save images from fixed noise
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        # Save images each epoch
        vutils.save_image(
            fake,
            os.path.join(CONFIG["sample_dir"], f"epoch_{epoch+1:03d}.png"),
            normalize=True,
        )

    # Save final models
    torch.save(
        netG.state_dict(), os.path.join(CONFIG["checkpoint_dir"], "generator.pth")
    )
    torch.save(
        netD.state_dict(), os.path.join(CONFIG["checkpoint_dir"], "discriminator.pth")
    )

    # Plot and save loss curves
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(CONFIG["output_dir"], "loss_plot.png"))
    plt.close()

    # Display training progression
    plt.figure(figsize=(15, 3))
    total_images = len(img_list)
    if total_images <= 5:
        selected_indices = list(range(total_images))
    else:
        # Select 5 evenly spaced images
        step = (total_images - 1) // 4
        selected_indices = [0] + [i * step for i in range(1, 5)]

    for i, idx in enumerate(selected_indices):
        plt.subplot(1, 5, i + 1)
        plt.axis("off")
        plt.title(f"Epoch {idx+1}")
        plt.imshow(np.transpose(img_list[idx], (1, 2, 0)))
    plt.savefig(os.path.join(CONFIG["output_dir"], "progress.png"))
    plt.close()

    # Display final results
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Final Generated Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.savefig(os.path.join(CONFIG["output_dir"], "final_result.png"))
    plt.close()

    print("Training complete!")


if __name__ == "__main__":
    main()
