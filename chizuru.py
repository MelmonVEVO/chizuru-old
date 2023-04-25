# This code is governed under the GNU General Public Licence v3.0.
#
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
# Some of the code here is inspired by the work of https://github.com/sebtheiler/tutorials/blob/main/dqn/train_dqn.py

"""This file contains everything needed to run the Chizuru AI."""

from rogue_gym.envs import RogueEnv
from random import choice, random
from collections import deque
import tensorflow as tf
import datetime
import numpy as np
import itertools

# Constants
ASCII_CHARNUM = 128
ENVIRONMENT = "rogueinabox"
EPISODES_PER_INTERVAL = 500
CKPT_PATH = "training/czr-{interval:04d}.ckpt"
LOG_DIR = "logs/czr" + datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
ACTIONS = ['h', 'j', 'k', 'l', 'u', 'n', 'b', 'y', 's', '.']  # Movement actions, search and wait.

# Hyperparameters
GAMMA = 0.99
NUM_ITERATIONS = 20000
MAX_TURNS_IN_EPISODE = 3000
BATCH_SIZE = 64
BUFFER_SIZE = 200000
MIN_REPLAY_SIZE = 2000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 150000
LEARNING_RATE = 0.00001
UPDATE_FREQUENCY = 10


class Agent:
    """Contains everything needed to manage the agent."""
    def __init__(self, h, w):
        self.online_net = create_dueling_dqn(h, w)
        self.target_net = create_dueling_dqn(h, w)

    def get_action(self, e, observation):
        """Agent chooses an action."""
        rnd_sample = random()

        if rnd_sample <= e:
            return choice(ACTIONS)
        else:
            return self.online_net.act(observation)

    def update_target_network(self):
        """Updates target network with the online network."""
        self.target_net.set_weights(self.online_net.get_weights())

    def learn(self, batch_size, gamma, turn_no):
        """Learns from replays."""
        pass


def create_dueling_dqn(h, w) -> tf.keras.Model:
    """Creates a Dueling DQN."""
    net_input = tf.keras.Input(shape=(h, w), dtype=tf.float32)

    conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(net_input)
    mp1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(mp1)
    mp2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(mp2)

    val, adv = tf.split(conv3, 2, 3)

    val = tf.keras.layers.Flatten()(val)
    val = tf.keras.layers.Dense(1)(val)

    adv = tf.keras.layers.Flatten()(adv)
    adv = tf.keras.layers.Dense(len(ACTIONS))(adv)

    reduced = tf.keras.layers.Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))

    output = tf.keras.layers.Add()([val, tf.keras.layers.Subtract()([adv, reduced(adv)])])

    final_model = tf.keras.Model(
        inputs=[input],
        outputs=[output]
    )

    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    return final_model


def create_rainbow_dqn(_h, _w):
    """Creates a Rainbow Deep Q-network."""
    pass


def save_checkpoint(model_sv: tf.keras.Model, interval) -> None:
    """Saves the model checkpoint with given interval."""
    model_sv.save_weights(CKPT_PATH.format(interval=interval))
    print("Episode " + str(interval) + " saved to " + CKPT_PATH.format(interval=interval) + "~")


def load_checkpoint(model_ld: tf.keras.Model, interval) -> tf.keras.Model:
    """Loads a model checkpoint at a given interval. Returns the loaded model."""
    model_ld.load_weights(CKPT_PATH)
    print("File " + CKPT_PATH.format(interval=interval) + " loaded to current model~")
    return model_ld


if __name__ == "__main__":
    agent = Agent(21, 79)

    tf.keras.utils.plot_model(agent.online_net, "stuff.png", show_shapes=True)

    writer = tf.summary.create_file_writer(LOG_DIR)

    replay_buffer = deque(maxlen=500)

    CONFIG = {
        'width': 79, 'height': 21,
        'dungeon': {
            'style': 'rogue',
            'room_num_x': 3, 'room_num_y': 2
        },
        'enemies': []
    }
    env = RogueEnv(max_steps=500, stair_reward=50.0, config_dict=CONFIG)
    episode_reward = 0
    turn = 0
    all_rewards = []
    all_losses = []
    env.reset()
    done = False
    state, _, _, _ = env.step('.')

    # Main processing
    try:
        with writer.as_default():
            for step in itertools.count():
                epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
                action = agent.get_action(epsilon, state)
                new_state, reward, done, _ = env.step(action)
                episode_reward += reward

                transition = (state, action, reward, done, new_state)
                replay_buffer.append(transition)
                state = new_state
                turn += 1

                # Learning step
                if turn % UPDATE_FREQUENCY == 0 and len(replay_buffer) > MIN_REPLAY_SIZE:
                    loss, _ = agent.learn(BATCH_SIZE, GAMMA, turn)

                if done:
                    env.reset()
                    all_rewards.append(episode_reward)
                    episode_reward = 0
                    turn = 0

    except KeyboardInterrupt:
        print("Exiting~")
        writer.close()

    env.close()


# †昇天†
