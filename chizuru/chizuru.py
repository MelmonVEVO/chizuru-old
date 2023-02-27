# This code is governed under the GNU General Public Licence v3.0.

"""This file contains the Chizuru class, a Rogue playing agent."""

import tensorflow as tf

import matplotlib.pyplot as plt

# Helper functions
def cnn_layer():
    return tf.keras.layers.Conv2D(
        50,
        (3, 3),
        activation="relu",

    )


class Chizuru:
    def __init__(self):
        pass

    def get_action(self, buffer):
        pass

    def learn(self):
        pass

    def save_checkpoint(self, filename="czr.ckpt"):
        pass

    def load_checkpoint(self, filename="czr.ckpt"):
        pass

    def save_agent(self, filename="czr.h5"):
        pass

    def load_agent(self, filename="czr.h5"):
        pass
