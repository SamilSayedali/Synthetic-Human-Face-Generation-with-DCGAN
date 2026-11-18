"""
Synthetic Human Face Generation with DCGAN
Author: Şamil Sayedali

This project trains a Deep Convolutional GAN (DCGAN) on a folder of face images.
The model learns to generate synthetic face-like images; it does not reproduce
or identify real people.

Expected dataset structure:
    dataset/
        image_001.jpg
        image_002.png
        ...

You can also point --data_dir to a directory containing class subfolders.
Images are loaded without labels.

Example:
    python train.py --data_dir ./dataset --epochs 100 --batch_size 32

For Kaggle, set DATA_DIR to the dataset's image directory, for example:
    /kaggle/input/face-mask-lite-dataset/without_mask
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# ----------------------------- Configuration -----------------------------

SEED = 42
IMAGE_SIZE = 64
CHANNELS = 3
LATENT_DIM = 128
AUTOTUNE = tf.data.AUTOTUNE

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ------------------------------- Data input -------------------------------

def load_dataset(data_dir: str, batch_size: int):
    """Load RGB images, resize them, and scale pixels to [-1, 1]."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_path.resolve()}\n"
            "Pass a valid directory using --data_dir."
        )

    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        label_mode=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        shuffle=True,
        seed=SEED,
        interpolation="bilinear",
    )

    def preprocess(images):
        images = tf.cast(images, tf.float32)
        return (images / 127.5) - 1.0

    dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


# -------------------------------- Generator --------------------------------

def build_generator():
    """Map random latent vectors to 64x64 RGB images in [-1, 1]."""
    inputs = tf.keras.Input(shape=(LATENT_DIM,), name="latent_vector")

    x = tf.keras.layers.Dense(4 * 4 * 512, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
    x = tf.keras.layers.Reshape((4, 4, 512))(x)

    for filters in (256, 128, 64):
        x = tf.keras.layers.Conv2DTranspose(
            filters, kernel_size=4, strides=2, padding="same", use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)

    outputs = tf.keras.layers.Conv2DTranspose(
        CHANNELS, kernel_size=4, strides=2, padding="same",
        activation="tanh", name="generated_image"
    )(x)

    return tf.keras.Model(inputs, outputs, name="generator")


# ------------------------------- Discriminator ----------------------------

def build_discriminator():
    """Return an unbounded real/fake logit (no sigmoid activation)."""
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

    x = inputs
    for filters in (64, 128, 256, 512):
        x = tf.keras.layers.Conv2D(
            filters, kernel_size=4, strides=2, padding="same"
        )(x)
        if filters != 64:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
        x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(1, name="real_fake_logit")(x)
    return tf.keras.Model(inputs, outputs, name="discriminator")


# ---------------------------- Losses and optimizers -----------------------

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_logits, fake_logits):
    real_loss = cross_entropy(tf.ones_like(real_logits), real_logits)
    fake_loss = cross_entropy(tf.zeros_like(fake_logits), fake_logits)
    return real_loss + fake_loss


def generator_loss(fake_logits):
    return cross_entropy(tf.ones_like(fake_logits), fake_logits)


# ------------------------------- Training ---------------------------------

@tf.function
def train_step(real_images, generator, discriminator,
               generator_optimizer, discriminator_optimizer):
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal((batch_size, LATENT_DIM))

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)

        real_logits = discriminator(real_images, training=True)
        fake_logits = discriminator(fake_images, training=True)

        gen_loss = generator_loss(fake_logits)
        disc_loss = discriminator_loss(real_logits, fake_logits)

    gen_gradients = gen_tape.gradient(
        gen_loss, generator.trainable_variables
    )
    disc_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gen_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(disc_gradients, discriminator.trainable_variables)
    )

    return gen_loss, disc_loss


def save_sample_images(generator, epoch, output_dir, count=16):
    """Save a fixed-noise grid so progress can be compared across epochs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fixed_noise = tf.random.stateless_normal(
        (count, LATENT_DIM), seed=(SEED, epoch)
    )
    predictions = generator(fixed_noise, training=False)
    predictions = (predictions + 1.0) / 2.0
    predictions = tf.clip_by_value(predictions, 0.0, 1.0).numpy()

    grid_size = int(np.ceil(np.sqrt(count)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    axes = np.array(axes).reshape(-1)

    for index, axis in enumerate(axes):
        axis.axis("off")
        if index < count:
            axis.imshow(predictions[index])

    fig.suptitle(f"Synthetic face samples — epoch {epoch}")
    fig.tight_layout()
    fig.savefig(output_path / f"samples_epoch_{epoch:04d}.png", dpi=150)
    plt.close(fig)


def train(data_dir, epochs, batch_size, output_dir):
    dataset = load_dataset(data_dir, batch_size)

    generator = build_generator()
    discriminator = build_discriminator()

    # Separate optimizers are used because the two networks are updated
    # independently in the adversarial training loop.
    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0002, beta_1=0.5
    )
    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0002, beta_1=0.5
    )

    generator.summary()
    discriminator.summary()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    history = {"generator_loss": [], "discriminator_loss": []}

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        gen_metric = tf.keras.metrics.Mean()
        disc_metric = tf.keras.metrics.Mean()

        for real_images in dataset:
            gen_loss_value, disc_loss_value = train_step(
                real_images,
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
            )
            gen_metric.update_state(gen_loss_value)
            disc_metric.update_state(disc_loss_value)

        gen_epoch_loss = float(gen_metric.result().numpy())
        disc_epoch_loss = float(disc_metric.result().numpy())
        history["generator_loss"].append(gen_epoch_loss)
        history["discriminator_loss"].append(disc_epoch_loss)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"G loss: {gen_epoch_loss:.4f} | "
            f"D loss: {disc_epoch_loss:.4f} | "
            f"time: {elapsed:.1f}s"
        )

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            save_sample_images(generator, epoch, output_path)

    # Save in the native Keras format.
    generator.save(output_path / "generator.keras")
    discriminator.save(output_path / "discriminator.keras")

    # Save training history and loss curve.
    np.save(output_path / "training_history.npy", history)
    plt.figure(figsize=(9, 5))
    plt.plot(history["generator_loss"], label="Generator")
    plt.plot(history["discriminator_loss"], label="Discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DCGAN training losses")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path / "loss_curve.png", dpi=150)
    plt.close()

    print(f"\nTraining complete. Results saved in: {output_path.resolve()}")
    return generator, discriminator, history


# ---------------------------------- CLI -----------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a DCGAN to generate synthetic face images."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Folder containing training images (or class subfolders)."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
