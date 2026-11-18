# Synthetic Human Face Generation with DCGAN

A TensorFlow/Keras implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) for learning from face-image datasets and generating synthetic face-like images.

## Important attribution note

This implementation is a rewritten and improved version based on the uploaded notebook's original GAN idea and workflow. Before publishing, compare it with the original source/notebook and cite any code, tutorial, or dataset that you reused. Only describe work as independently developed to the extent that you actually implemented and understand it.

## Features

- 64×64 RGB image pipeline normalized to `[-1, 1]`
- Convolutional Generator and Discriminator
- Logit-based binary cross-entropy loss (`from_logits=True`)
- Separate Adam optimizers for the two networks
- Reproducible random seed
- Sample grids saved during training
- Saved Keras models and loss curve

## Dataset

Place face images in a directory, for example:

```text
dataset/
  image_001.jpg
  image_002.png
  ...
```

The loader also supports class subdirectories. Use only datasets whose licenses permit your intended use. The original notebook used the `face-mask-lite-dataset` `without_mask` directory on Kaggle; this script does not download that dataset automatically.

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py --data_dir ./dataset --epochs 100 --batch_size 32
```

For Kaggle, for example:

```bash
python train.py --data_dir /kaggle/input/face-mask-lite-dataset/without_mask --epochs 100 --batch_size 32
```

Training time depends on dataset size and hardware. A GPU is recommended.

## Outputs

The `outputs/` directory contains generated image grids, `generator.keras`, `discriminator.keras`, `training_history.npy`, and `loss_curve.png`.

## Limitations

This is a baseline DCGAN, not a state-of-the-art face generator. Image quality depends heavily on dataset quality, dataset size, training duration, and hardware. Generated images may contain visual artifacts and should be labelled as AI-generated.
