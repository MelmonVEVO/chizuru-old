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
# Much of the code here is inspired by the work of https://github.com/sebtheiler/tutorials/blob/main/dqn/train_dqn.py,
# especially the code for agent learning.

"""This file contains everything needed to run the chizuru-rogue AI."""

from rogue_gym.envs import RogueEnv
from random import random, sample, randint
from collections import deque
import tensorflow as tf
import datetime
import numpy as np
import itertools

# Constants
STEPS_PER_INTERVAL = 10000
CKPT_PATH = "training/czr-{interval:04d}-{label}.ckpt"
LOG_DIR = "logs/czr" + datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
ACTIONS = ['h', 'j', 'k', 'l', 'u', 'n', 'b', 'y', 's', '.']  # Movement actions, search and wait.

# Hyperparameters
GAMMA = 0.99
NUM_ITERATIONS = 20000
MAX_TURNS_IN_EPISODE = 1000
BATCH_SIZE = 32
BUFFER_SIZE = 200000
MIN_REPLAY_SIZE = 1500
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 150000
LEARNING_RATE = 0.00001
UPDATE_FREQUENCY = 1000


class Agent:
    """Contains everything needed to manage the agent."""
    def __init__(self, h, w):
        self.h = h
        self.w = w
        self.online_net = create_dueling_dqn(h, w)
        self.target_net = create_dueling_dqn(h, w)
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)

    def get_action(self, s, e):
        """Agent chooses an action."""
        rnd_sample = random()

        if rnd_sample <= e:
            return randint(0, len(ACTIONS)-1)
        return self.online_net.predict(tf.reshape(tf.convert_to_tensor(s), (-1, 21, 79, 1)))[0].argmax()

    def update_target_network(self):
        """Updates target network with the online network."""
        self.target_net.set_weights(self.online_net.get_weights())

    def learn(self, batch_size, gamma):  # god, I'm so tired.
        """Learns from replays."""
        minibatch = sample(self.replay_buffer, BATCH_SIZE)

        states = tf.constant([r[0] for r in minibatch])
        actions = tf.constant([r[1] for r in minibatch])
        rewards = tf.constant([r[2] for r in minibatch])
        dones = tf.constant([r[3] for r in minibatch])
        new_states = tf.constant([r[4] for r in minibatch])

        arg_q_max = self.online_net.predict(tf.reshape(new_states, (batch_size, 21, 79, 1))).argmax(axis=1)

        future_q_vals = self.target_net.predict(tf.reshape(new_states, (batch_size, 21, 79, 1)))
        double_q = future_q_vals[range(batch_size), arg_q_max]

        target_q = tf.cast(rewards, tf.float32) + (gamma * double_q * (1.0 - tf.cast(dones, tf.float32)))

        with tf.GradientTape() as tape:
            q_values = self.online_net(tf.reshape(states, (batch_size, 21, 79, 1)))

            one_hot_actions = tf.keras.utils.to_categorical(actions, len(ACTIONS), dtype=np.float32)
            q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)

            error = q - target_q
            learn_loss = tf.keras.losses.Huber()(target_q, q)

            model_gradients = tape.gradient(learn_loss, self.online_net.trainable_variables)
            self.online_net.optimizer.apply_gradients(zip(model_gradients, self.online_net.trainable_variables))

        return float(learn_loss.numpy()), error

    def save(self, interval):
        """Saves model at current interval."""
        save_checkpoint(self.online_net, intr, "online")
        save_checkpoint(self.target_net, interval, "target")

    def load(self, interval):
        """Loads model at given interval."""
        self.online_net = load_checkpoint(self.online_net, intr, "online")
        self.target_net = load_checkpoint(self.target_net, interval, "online")


def create_dueling_dqn(h, w) -> tf.keras.Model:
    """Creates a Dueling DQN."""
    net_input = tf.keras.Input(shape=(h, w, 1))
    net_input = tf.keras.layers.Lambda(lambda layer: layer / 255)(net_input)

    conv1 = tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation="relu")(net_input)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu")(conv1)
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu")(conv2)

    val, adv = tf.keras.layers.Lambda(lambda ww: tf.split(ww, 2, 3))(conv3)

    val = tf.keras.layers.Flatten()(val)
    val = tf.keras.layers.Dense(1)(val)

    adv = tf.keras.layers.Flatten()(adv)
    adv = tf.keras.layers.Dense(len(ACTIONS))(adv)

    reduced = tf.keras.layers.Lambda(lambda ww: tf.reduce_mean(ww, axis=1, keepdims=True))

    output = tf.keras.layers.Add()([val, tf.keras.layers.Subtract()([adv, reduced(adv)])])

    final_model = tf.keras.Model(net_input, output)

    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    return final_model


def create_rainbow_dqn(_h, _w):
    """Creates a Rainbow Deep Q-network."""
    pass


def save_checkpoint(model_sv: tf.keras.Model, interval, label) -> None:
    """Saves the model checkpoint with given interval."""
    model_sv.save_weights(CKPT_PATH.format(interval=interval, label=label))
    print("Interval " + str(interval) + " saved to " + CKPT_PATH.format(interval=interval, label=label) + "~")


def load_checkpoint(model_ld: tf.keras.Model, interval, label) -> tf.keras.Model:
    """Loads a model checkpoint at a given interval. Returns the loaded model."""
    model_ld.load_weights(CKPT_PATH.format(interval=interval, label=label))
    print("File " + CKPT_PATH.format(interval=interval, label=label) + " loaded to current model~")
    return model_ld


if __name__ == "__main__":
    agent = Agent(21, 79)

    writer = tf.summary.create_file_writer(LOG_DIR)

    CONFIG = {
        'width': 79, 'height': 21,
        'dungeon': {
            'style': 'rogue',
            'room_num_x': 3, 'room_num_y': 2
        },
        'enemies': []
    }
    env = RogueEnv(max_steps=MAX_TURNS_IN_EPISODE, stair_reward=100.0, config_dict=CONFIG)
    episode_reward = 0
    intr = 0
    episode = 0
    all_rewards = []
    all_losses = []
    state = env.reset()
    new_state, reward, done, _ = env.step('.')
    for _ in range(4):
        agent.replay_buffer.append((state.gray_image(), 9, reward, done, new_state.gray_image()))
    state = new_state

    # Main processing
    try:
        with writer.as_default():
            for step in itertools.count():
                epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
                action = agent.get_action(state.gray_image(), epsilon)
                new_state, reward, done, _ = env.step(ACTIONS[action])
                episode_reward += reward
                all_rewards.append(reward)

                transition = (state.gray_image(), action, reward, done, new_state.gray_image())
                agent.replay_buffer.append(transition)
                state = new_state

                # Learning step
                if step % UPDATE_FREQUENCY == 0 and len(agent.replay_buffer) > MIN_REPLAY_SIZE:
                    loss, _ = agent.learn(BATCH_SIZE, GAMMA)
                    all_losses.append(loss)

                if step % UPDATE_FREQUENCY == 0 and step > MIN_REPLAY_SIZE:
                    agent.update_target_network()

                if done:
                    dlvl = state.dungeon_level
                    env.reset()
                    all_rewards.append(episode_reward)
                    tf.summary.scalar('Evaluation score', episode_reward, step)
                    tf.summary.scalar('Dungeon level', dlvl, step)
                    print('\nEpisode', episode)
                    print('Reward this game', episode_reward)
                    print('Average reward this session', np.mean(all_rewards))
                    print('Epsilon', epsilon)
                    episode_reward = 0
                    episode += 1

                if step % STEPS_PER_INTERVAL == 0 and step > 0:
                    print('\nInterval', intr)
                    agent.save(intr)
                    intr += 1

    except KeyboardInterrupt:
        print("Exiting~")
        writer.close()
    env.close()


# †昇天†
