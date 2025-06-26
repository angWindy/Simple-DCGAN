import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import CONFIG


class CelebaDataset(Dataset):
    def __init__(self, root_dir=None, transform=None):
        # Sử dụng đường dẫn từ CONFIG nếu không có root_dir
        self.root_dir = root_dir if root_dir is not None else CONFIG["celeba_dir"]
        self.transform = transform

        # Kiểm tra thư mục tồn tại
        if not os.path.exists(self.root_dir):
            raise RuntimeError(
                f"Dataset directory not found: {self.root_dir}. Please run download_data.py first."
            )

        self.image_files = [
            f
            for f in os.listdir(self.root_dir)
            if f.endswith((".jpg", ".png", ".jpeg"))
        ]

        print(f"Found {len(self.image_files)} images in {self.root_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image, 0  # Return 0 as dummy label


def get_transforms():
    return transforms.Compose(
        [
            transforms.Resize(CONFIG["img_size"]),
            transforms.CenterCrop(CONFIG["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # Normalize from -1 to 1
        ]
    )
