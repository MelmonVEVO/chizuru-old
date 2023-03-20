# This code is governed under the GNU General Public Licence v3.0.

"""This file contains the Chizuru class, a Rogue playing agent."""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

ASCII_CHARNUM = 128

NUM_ITERATIONS = 20000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LOG_INTERVAL = 200

ENVIRONMENT = "rogueinabox"

map_net = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(21, 79)),
        tf.keras.layers.Embedding(ASCII_CHARNUM, 64, input_length=21 * 79),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(21, 79)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu")
    ]
)

crop_net = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(9, 9)),
        tf.keras.layers.Embedding(ASCII_CHARNUM, 64, input_length=9 * 9),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(9, 9)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu")
    ]
)
status_net = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(40,)),
        tf.keras.layers.Embedding(ASCII_CHARNUM, 64, inputlength=40),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu")
    ]
)

inv_net = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(60,)),
        tf.keras.layers.Embedding(ASCII_CHARNUM, 64, input_length=60),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
    ]
)


def get_crop(map):
    pass


class MapNet(tf.keras.Model):
    def __init__(self, h, w):
        super().__init__()


class Chizuru(tf.keras.Model):
    def __init__(self):
        super().__init__()
        inputs = tf.keras.Input(shape=())

    def get_action(self, state, buffer):
        pass

    def learn(self):
        pass

    def save_checkpoint(self, filename="czr.ckpt"):
        pass

    def load_checkpoint(self, filename="czr.ckpt"):
        pass

    def save_agent(self, filename="czr.h5"):
        self.save(filename)
