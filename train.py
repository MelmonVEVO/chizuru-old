#!/usr/bin/env python

# This code is governed under the GNU General Public Licence v3.0.

"""Runs and trains the Chizuru agent again and again until the program is halted or until
epoch count reaches a provided number."""

from chizuru import Chizuru
import tensorflow as tf
import datetime

# Globals
LOG_DIR = "logs/czr" + datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
TENSORBOARD_CALLBACK = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

# Hyperparameters
NUM_ITERATIONS = 20000
BATCH_SIZE = 32
BUFFER_SIZE = 200000
MIN_REPLAY_SIZE = 2000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 150000
TARGET_UPDATE_FREQUENCY = 10000
# End of hyperparameters

if __name__ == "__main__":
    pass
