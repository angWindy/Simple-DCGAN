# Simple DCGAN for Face Generation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Download Dataset

```bash
python download_data.py
```

This will download the CelebA-HQ dataset from Kaggle and set up the correct path.

### Step 2: Train Model

```bash
python train.py
```

### Step 3: Generate Images

```bash
python generate.py
```

## Project Structure

- `config.py`: Configuration parameters
- `models/`: Neural network model definitions
- `datasets/`: Dataset handling code
- `utils/`: Utility functions
- `train.py`: Training script
- `generate.py`: Image generation script
- `download_data.py`: Dataset downloader

## Results

Generated images will be saved to the `samples/` directory, with a progression image in `output/progress.png`.
