# easy_wgan
A simplified version of a WGAN (Wasserstein Generative Adversarial Network) image generation network built using PyTorch

```markdown
# Wasserstein Generative Adversarial Network (WGAN) in PyTorch

This repository contains a PyTorch implementation of a Wasserstein Generative Adversarial Network (WGAN) for image generation. WGAN is known for its stability and improved training dynamics compared to traditional GANs.

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python (>=3.6)
- PyTorch (>=1.0)
- torchvision
- NumPy
- argparse

You can install the required packages using `pip`:

```bash
pip install torch torchvision numpy argparse
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/ZRliAlex/easy_wgan.git
cd easy_wgan
```

2. Prepare your dataset:
   - Place your dataset in the `img` folder or specify the path to your dataset using the `--data_dir` flag.

3. Run the training script:

```bash
python wgan.py
```

### Training Options

- `--batch_size`: Batch size (default: 512)
- `--nz`: Size of the latent z vector (default: 96)
- `--ngf`: Number of generator filters (default: 96)
- `--ndf`: Number of discriminator filters (default: 96)
- `--lr`: Learning rate (default: 0.0002)
- `--n_epochs`: Number of training epochs (default: 5000)
- `--image_size`: Size of generated images (default: 96)
- `--lambda_gp`: Weight of gradient penalty (default: 10)
- `--n_critic`: Number of critic iterations per generator iteration (default: 5)
- `--save_interval`: Interval for saving generated images (default: 5)
- `--save_model_interval`: Interval for saving model checkpoints (default: 200)
- `--clip_value`: Gradient clip (default: 1)

### Model Functions

All model functions are saved in `model.py`. You can find the architecture details and configurations for the generator and discriminator there.

### Dataset Cleaning

To clean your datasets, you can use `img_ded.py`. This script is designed to assist in dataset cleaning and preprocessing.

### Results

Generated images will be saved in the `result/` directory. Model checkpoints will be saved in the `pth/` directory.

## Acknowledgments

This code is based on the WGAN paper by Martin Arjovsky et al. (https://arxiv.org/abs/1701.07875). 
