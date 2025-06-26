import os
import shutil
import sys

import kagglehub

# Thêm đường dẫn gốc vào sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG


def download_dataset():
    """
    Tải dataset CelebA-HQ từ Kaggle và lưu vào thư mục data trong project
    """
    print("Downloading CelebA-HQ dataset...")

    # Tạo thư mục data
    os.makedirs(CONFIG["data_dir"], exist_ok=True)

    # Tải dataset từ Kaggle
    kaggle_path = kagglehub.dataset_download("badasstechie/celebahq-resized-256x256")
    print(f"Dataset downloaded to: {kaggle_path}")

    # Đường dẫn đến thư mục chứa ảnh
    celeba_source_dir = os.path.join(kaggle_path, "celeba_hq_256")
    if not os.path.exists(celeba_source_dir):
        print(f"Warning: Could not find 'celeba_hq_256' directory in {kaggle_path}")
        print("Available directories:")
        print(os.listdir(kaggle_path))
        return False

    # Copy dữ liệu vào thư mục project
    if not os.path.exists(CONFIG["celeba_dir"]):
        print(f"Copying dataset to project directory: {CONFIG['celeba_dir']}")
        shutil.copytree(celeba_source_dir, CONFIG["celeba_dir"])
    else:
        print(f"Dataset already exists at: {CONFIG['celeba_dir']}")

    # Đếm số lượng ảnh
    image_count = len(
        [
            f
            for f in os.listdir(CONFIG["celeba_dir"])
            if f.endswith((".jpg", ".png", ".jpeg"))
        ]
    )
    print(f"Dataset contains {image_count} images")

    return True


if __name__ == "__main__":
    success = download_dataset()
    if success:
        print("Dataset ready for training!")
    else:
        print("Failed to prepare dataset.")
