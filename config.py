import os

# Lấy đường dẫn gốc của project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # Data
    "data_dir": os.path.join(
        PROJECT_ROOT, "data"
    ),  # Thư mục chứa dữ liệu trong project
    "celeba_dir": os.path.join(
        PROJECT_ROOT, "data", "celeba_hq_256"
    ),  # Thư mục chứa ảnh CelebA
    "workers": 2,
    "batch_size": 128,
    "img_size": 64,
    "nc": 3,  # Number of channels
    "nz": 100,  # Size of z latent vector
    "ngf": 64,  # Size of feature maps in generator
    "ndf": 64,  # Size of feature maps in discriminator
    # Training
    "num_epochs": 5,
    "learning_rate": 2e-4,
    "weight_decay": 1e-5,  # Weight decay for AdamW Optimizers
    "beta1": 0.5,  # Beta1 for Adam optimizer
    # Output
    "output_dir": os.path.join(PROJECT_ROOT, "output"),
    "checkpoint_dir": os.path.join(PROJECT_ROOT, "checkpoints"),
    "sample_dir": os.path.join(PROJECT_ROOT, "samples"),
}
