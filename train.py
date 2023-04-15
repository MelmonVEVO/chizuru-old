#!/usr/bin/env python3

# This code is governed under the GNU General Public Licence v3.0.

"""Runs and trains the Chizuru agent again and again until the program is halted or until
epoch count reaches a provided number."""

from chizuru import create_model, save_checkpoint, load_checkpoint, get_crop
import tensorflow as tf
import datetime
import os

# Constants
LOG_DIR = "logs/czr" + datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
TENSORBOARD_CALLBACK = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
CKPT_PATH = "training/czr-{epoch:04d}.ckpt"
CKPT_DIR = os.path.dirname(CKPT_PATH)
BATCH_SIZE = 64
CKPT_CALLBACK = tf.keras.callbacks.ModelCheckpoint(
    filepath=CKPT_PATH,
    save_weights_only=True,
    verbose=1,
    save_freq=5*BATCH_SIZE
)

# Hyperparameters
NUM_ITERATIONS = 20000
BATCH_SIZE = 32
BUFFER_SIZE = 200000
MIN_REPLAY_SIZE = 2000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 150000
TARGET_UPDATE_FREQUENCY = 10000
ALPHA = 1.0e-3
BETA1 = 0.9
BETA2 = 0.999


if __name__ == "__main__":
    pass
