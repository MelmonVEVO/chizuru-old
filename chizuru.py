# This code is governed under the GNU General Public Licence v3.0.

#   ██████╗██╗  ██╗██╗███████╗██╗   ██╗██████╗ ██╗   ██╗
#  ██╔════╝██║  ██║██║╚══███╔╝██║   ██║██╔══██╗██║   ██║
#  ██║     ███████║██║  ███╔╝ ██║   ██║██████╔╝██║   ██║
#  ██║     ██╔══██║██║ ███╔╝  ██║   ██║██╔══██╗██║   ██║
#  ╚██████╗██║  ██║██║███████╗╚██████╔╝██║  ██║╚██████╔╝
#   ╚═════╝╚═╝  ╚═╝╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝
#  ██████╗  ██████╗  ██████╗ ██╗   ██╗███████╗
#  ██╔══██╗██╔═══██╗██╔════╝ ██║   ██║██╔════╝
#  ██████╔╝██║   ██║██║  ███╗██║   ██║█████╗
#  ██╔══██╗██║   ██║██║   ██║██║   ██║██╔══╝
#  ██║  ██║╚██████╔╝╚██████╔╝╚██████╔╝███████╗
#  ╚═╝  ╚═╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚══════╝
#
# An AI that plays Rogue.

"""This file contains everything needed to run the Chizuru AI."""

import os
import tensorflow as tf

ASCII_CHARNUM = 128
ENVIRONMENT = "rogueinabox"
LOG_INTERVAL = 200

CKPT_PATH = "training/czr-{epoch:04d}.ckpt"
CKPT_DIR = os.path.dirname(CKPT_PATH)

# Hyperparameters
NUM_ITERATIONS = 20000
BATCH_SIZE = 64
ALPHA = 1.0e-3
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1.0e-8
DECAY = 0.0

CKPT_CALLBACK = tf.keras.callbacks.ModelCheckpoint(
    filepath=CKPT_PATH,
    save_weights_only=True,
    verbose=1,
    save_freq=5*BATCH_SIZE
)


def create_model():
    """Instantiates, compiles and returns the Chizuru model."""
    status_input = tf.keras.Input(shape=(64,))
    inv_input = tf.keras.Input(shape=(64,))
    equip_input = tf.keras.Input(shape=(64,))
    map_input = tf.keras.Input(shape=(21, 79), dtype=tf.int32)
    crop_input = tf.keras.Input(shape=(9, 9), dtype=tf.int32)

    status_net = tf.keras.layers.Embedding(ASCII_CHARNUM, 64)(status_input)
    status_net = tf.keras.layers.Dense(32, activation="relu")(status_net)
    status_net = tf.keras.layers.Dense(32, activation="relu")(status_net)
    status_net = tf.keras.layers.Dense(16, activation="relu")(status_net)

    inv_net = tf.keras.layers.Embedding(ASCII_CHARNUM, 64)(inv_input)
    inv_net = tf.keras.layers.Dense(32, activation="relu")(inv_net)
    inv_net = tf.keras.layers.Dense(16, activation="relu")(inv_net)

    equip_net = tf.keras.layers.Embedding(ASCII_CHARNUM, 16)(equip_input)
    equip_net = tf.keras.layers.Dense(32, activation="relu")(equip_net)
    equip_net = tf.keras.layers.Dense(16, activation="relu")(equip_net)

    map_net = tf.keras.layers.Embedding(ASCII_CHARNUM, 64, input_length=21 * 79)(map_input)
    map_net = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(21, 79))(map_net)
    map_net = tf.keras.layers.MaxPooling2D((2, 2))(map_net)
    map_net = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(map_net)
    map_net = tf.keras.layers.MaxPooling2D((2, 2))(map_net)
    map_net = tf.keras.layers.Conv2D(16, (3, 3), activation="relu")(map_net)

    crop_net = tf.keras.layers.Embedding(ASCII_CHARNUM, 64, input_length=9 * 9)(crop_input)
    crop_net = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(9, 9))(crop_net)
    crop_net = tf.keras.layers.MaxPooling2D((2, 2))(crop_net)
    crop_net = tf.keras.layers.Conv2D(16, (3, 3), activation="relu")(crop_net)

    # requires inputs with matching shapes except for the concatenation axis. Received: input_shape=[(None, 16, 16), (None, 64, 16), (None, 16, 16), (None, 1, 16, 64), (None, 1, 1, 32)]
    collected = tf.keras.layers.Concatenate()([status_net, inv_net, equip_net, map_net, crop_net])

    # MLP after concat
    pre_mlp = tf.keras.layers.Dense(64, activation="relu")(collected)
    pre_mlp = tf.keras.layers.Dense(64, activation="relu")(pre_mlp)
    pre_mlp = tf.keras.layers.Dense(48, activation="relu")(pre_mlp)

    # LSTM
    lstm = tf.keras.layers.LSTM(128)(pre_mlp)

    # final MLP
    final_mlp = tf.keras.layers.Dense(128)(lstm)
    final_mlp = tf.keras.layers.Dense(64)(final_mlp)

    output = tf.keras.layers.Dense(10)(final_mlp)

    final_model = tf.keras.Model(
        inputs=[status_input,
                inv_input,
                equip_input,
                map_input,
                crop_input],
        outputs=[output]
    )

    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    return final_model


def get_crop(map: list[list[int]]):  # TODO
    """Returns a 9x9 crop of the given Rogue map surrounding the player."""
    pass


def save_checkpoint(model_sv: tf.keras.Model, epoch):
    """Saves the model checkpoint with given epoch."""
    model_sv.save_weights(CKPT_PATH.format(epoch=epoch))
    print("Epoch " + str(epoch) + " saved to " + CKPT_PATH.format(epoch=epoch) + "~")


def load_checkpoint(model_ld: tf.keras.Model, epoch):
    """Loads a model checkpoint at a given epoch."""
    model_ld.load_weights(CKPT_PATH)
    print("File " + CKPT_PATH.format(epoch=epoch) + " loaded to current model~")


if __name__ == "__main__":
    model = create_model()
    tf.keras.utils.plot_model(model, "stuff.png", show_shapes=True)
    save_checkpoint(model, 0)


# †昇天†
