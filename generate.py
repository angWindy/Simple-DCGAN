import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils

from config import CONFIG
from models import Generator


def main():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create output directory
    os.makedirs("generated_images", exist_ok=True)

    # Load model
    model_path = os.path.join(CONFIG["checkpoint_dir"], "generator.pth")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # Initialize model
    netG = Generator().to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()

    # Generate images
    num_images = 16
    noise = torch.randn(num_images, CONFIG["nz"], 1, 1, device=device)

    with torch.no_grad():
        generated_images = netG(noise).detach().cpu()

    # Save grid of images
    grid = vutils.make_grid(generated_images, padding=2, normalize=True, nrow=4)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.savefig("generated_images/grid.png")

    # Save individual images
    for i in range(num_images):
        vutils.save_image(
            generated_images[i], f"generated_images/img_{i+1:03d}.png", normalize=True
        )

    print(f"Generated {num_images} images in 'generated_images' directory")


if __name__ == "__main__":
    main()
