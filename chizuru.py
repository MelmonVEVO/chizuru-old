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

# ****************************************************************************************************** #
# The following code was adapted from:                                                                   #
# Author: Sebastian Theiler                                                                              #
# Accessed from: https://github.com/sebtheiler/tutorials/blob/main/dqn/train_dqn.py                      #
# Date of last retrieval: 26-04-2023                                                                     #
# ****************************************************************************************************** #

"""This file contains everything needed to run the chizuru-rogue AI."""

from rogue_gym.envs import RogueEnv
from random import randint
import tensorflow as tf
import datetime
import numpy as np
import itertools
import argparse
import os

# Constants
STEPS_PER_INTERVAL = 10000
CKPT_PATH = "training/czr-{interval:04d}-{label}.ckpt"
LOG_DIR = "logs/czr" + datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
ACTIONS = ['h', 'j', 'k', 'l', 'u', 'n', 'b', 'y', 's', '.']  # Movement actions, search and wait.
PARSER = argparse.ArgumentParser(description="Interval checkpoint to load from.")
PARSER.add_argument('-i', '--interval', action="store")
HISTORY_LEN = 4

# Hyperparameters
GAMMA = 0.99
NUM_ITERATIONS = 20000
MAX_TURNS_IN_EPISODE = 1500
BATCH_SIZE = 32
BUFFER_SIZE = 100000
MIN_REPLAY_SIZE = 1500
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 150000
LEARNING_RATE = 0.00001
TARGET_UPDATE_FREQUENCY = 750
PRIORITY_SCALE = 0.7


class ReplayBuffer:
    """ReplayBuffer for storing transitions.
    This implementation was heavily inspired by Fabio M. Graetz's replay buffer
    here: https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb"""

    def __init__(self, size=BUFFER_SIZE, input_shape=(21, 79),
                 history_length=HISTORY_LEN):  # History length for n-step learning
        """
        Arguments:
            size: Integer, Number of stored transitions
            input_shape: Shape of the preprocessed frame
            history_length: Integer, Number of frames stacked together to create a state for the agent
        """
        self.size = size
        self.input_shape = input_shape
        self.history_length = history_length
        self.count = 0  # total index of memory written to, always less than self.size
        self.current = 0  # index to write to

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.uint8)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.input_shape[0], self.input_shape[1]), dtype=np.float32)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        self.priorities = np.zeros(self.size, dtype=np.float32)

    def add_experience(self, action, frame, reward, terminal):
        """Saves a transition to the replay buffer
        Arguments:
            action: An integer between 0 and env.action_space.n - 1
                determining the action the agent perfomed
            frame: A (21, 74, 1) frame of the game in grayscale
            reward: A float determining the reward the agend received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != self.input_shape:
            raise ValueError('Dimension of frame is wrong!')

        # Write memory
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.priorities[self.current] = max(self.priorities.max(initial=0),
                                            1)  # make the most recent experience important
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def get_minibatch(self, batch_size=32, priority_scale=0.0):
        """Returns a minibatch of self.batch_size = 32 transitions
        Arguments:
            batch_size: How many samples to return
            priority_scale: How much to weight priorities. 0 = completely random, 1 = completely based on priority
        Returns:
            A tuple of states, actions, rewards, new_states, and terminals
            If use_per is True:
                An array describing the importance of transition. Used for scaling gradient steps.
                An array of each index that was sampled
        """

        if self.count < self.history_length:
            raise ValueError('Not enough memories to get a minibatch')

        # Get sampling probabilities from priority list
        scaled_priorities = self.priorities[self.history_length:self.count - 1] ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)

        # Get a list of valid indices
        indices = []
        for i in range(batch_size):
            while True:
                # Get a random number from history_length to maximum frame written with probabilities based on
                # priority weights
                index = np.random.choice(np.arange(self.history_length, self.count - 1), p=sample_probabilities)

                # We check that all frames are from same episode with the two following if statements.  If either are
                # True, the index is invalid.
                if index >= self.current >= index - self.history_length:
                    continue
                if self.terminal_flags[index - self.history_length:index].any():
                    continue
                break
            indices.append(index)

        # Retrieve states from memory
        states = []
        new_states = []
        for idx in indices:
            states.append(self.frames[idx - self.history_length:idx, ...])
            new_states.append(self.frames[idx - self.history_length + 1:idx + 1, ...])

        states = np.transpose(np.asarray(states), axes=(0, 2, 3, 1))
        new_states = np.transpose(np.asarray(new_states), axes=(0, 2, 3, 1))

        # Get importance weights from probabilities calculated earlier
        importance = 1 / self.count * 1 / sample_probabilities[[index - self.history_length for index in indices]]
        importance = importance / importance.max()

        return (states, self.actions[indices], self.rewards[indices], new_states, self.terminal_flags[indices]), importance, indices

    def set_priorities(self, indices, errors, offset=0.1):
        """Update priorities for PER
        Arguments:
            indices: Indices to update
            errors: For each index, the error between the target Q-vals and the predicted Q-vals
            offset
        """
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def save(self, folder_name):
        """Save the replay buffer to a folder"""

        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        np.save(folder_name + '/actions.npy', self.actions)
        np.save(folder_name + '/frames.npy', self.frames)
        np.save(folder_name + '/rewards.npy', self.rewards)
        np.save(folder_name + '/terminal_flags.npy', self.terminal_flags)

    def load(self, folder_name):
        """Loads the replay buffer from a folder"""
        self.actions = np.load(folder_name + '/actions.npy')
        self.frames = np.load(folder_name + '/frames.npy')
        self.rewards = np.load(folder_name + '/rewards.npy')
        self.terminal_flags = np.load(folder_name + '/terminal_flags.npy')


class Agent:
    """Contains everything needed to manage the agent."""

    def __init__(self, h, w):
        self.h = h
        self.w = w
        self.online_net = create_dueling_dqn(h, w)
        self.target_net = create_dueling_dqn(h, w)
        self.replay_buffer = ReplayBuffer()

    def get_action(self, s, e):
        """Agent chooses an action."""
        rnd_sample = np.random.rand()

        if rnd_sample <= e:
            return randint(0, len(ACTIONS) - 1)
        reshaped = s.reshape((1, 21, 79, 4))
        q_vals = self.online_net.predict(reshaped)[0]
        return q_vals.argmax()

    def update_target_network(self):
        """Updates target network with the online network."""
        self.target_net.set_weights(self.online_net.get_weights())

    def learn(self, batch_size, gamma, e, priority_scale=1.0):  # god, I'm so tired.
        """Learns from replays."""
        (states, actions, rewards, new_states, dones), importance, indices = self.replay_buffer.get_minibatch(
            batch_size=batch_size, priority_scale=priority_scale)
        importance = importance ** (1 - e)

        arg_q_max = self.online_net.predict(new_states).argmax(axis=1)

        future_q_vals = self.target_net.predict(new_states)
        double_q = future_q_vals[range(batch_size), arg_q_max]

        target_q = rewards + (gamma * double_q * (1.0 - dones))

        with tf.GradientTape() as tape:
            q_values = self.online_net(states)

            one_hot_actions = tf.keras.utils.to_categorical(actions, len(ACTIONS), dtype=np.float32)
            q = tf.reduce_sum(tf.multiply(q_values, one_hot_actions), axis=1)

            error = q - target_q
            learn_loss = tf.keras.losses.MeanSquaredError()(target_q, q)
            learn_loss = tf.reduce_mean(learn_loss * importance)

            model_gradients = tape.gradient(learn_loss, self.online_net.trainable_variables)
            self.online_net.optimizer.apply_gradients(zip(model_gradients, self.online_net.trainable_variables))

            self.replay_buffer.set_priorities(indices, error)

        return float(learn_loss.numpy()), error

    def save(self, interval):
        """Saves model at current interval."""
        save_checkpoint(self.online_net, interval, "online")
        save_checkpoint(self.target_net, interval, "target")

    def load(self, interval):
        """Loads model at given interval."""
        self.online_net = load_checkpoint(self.online_net, interval, "online")
        self.target_net = load_checkpoint(self.target_net, interval, "target")


def create_dueling_dqn(h, w) -> tf.keras.Model:
    """Creates a Dueling DQN."""
    net_input = tf.keras.Input(shape=(h, w, HISTORY_LEN))
    # net_input = tf.keras.layers.Lambda(lambda layer: layer / 255)(net_input)

    conv1 = tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation="relu", use_bias=False,
                                   kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.))(net_input)
    conv2 = tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu", use_bias=False,
                                   kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.))(conv1)
    conv3 = tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu", use_bias=False,
                                   kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.))(conv2)
    noise = tf.keras.layers.GaussianNoise(0.1)(conv3)

    val, adv = tf.keras.layers.Lambda(lambda ww: tf.split(ww, 2, 3))(noise)

    val = tf.keras.layers.Flatten()(val)
    val = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.))(val)

    adv = tf.keras.layers.Flatten()(adv)
    adv = tf.keras.layers.Dense(len(ACTIONS), kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.))(adv)

    reduced = tf.keras.layers.Lambda(lambda ww: tf.reduce_mean(ww, axis=1, keepdims=True))

    output = tf.keras.layers.Add()([val, tf.keras.layers.Subtract()([adv, reduced(adv)])])

    final_model = tf.keras.Model(net_input, output)

    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.MeanSquaredError()
    )

    return final_model


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

    arg = PARSER.parse_args()

    if arg.interval:
        agent.load(arg.interval)

    writer = tf.summary.create_file_writer(LOG_DIR)

    CONFIG = {
        'width': 79,
        'height': 21,
        'hide_dungeon': True,
        'dungeon': {
            'style': 'rogue',
            'room_num_x': 3,
            'room_num_y': 2,
        },
        'enemies': {
            'enemies': []
        }
    }
    env = RogueEnv(max_steps=MAX_TURNS_IN_EPISODE, stair_reward=100.0, config_dict=CONFIG)
    episode_reward = 0
    intr = 0
    episode = 0
    all_rewards = []
    interval_rewards = []
    all_losses = []
    env.reset()
    new_state, rew, done, _ = env.step('.')
    agent.replay_buffer.add_experience(9, new_state.gray_image()[0], rew, done)
    current_game_state = np.repeat(new_state.gray_image().reshape(21, 79, 1), HISTORY_LEN, axis=2)  # with a history

    # Main processing
    try:
        with writer.as_default():
            for step in itertools.count():
                epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
                act = agent.get_action(current_game_state, epsilon)
                new_state, rew, done, _ = env.step(ACTIONS[act])
                episode_reward += rew
                all_rewards.append(rew)
                interval_rewards.append(rew)
                all_rewards = all_rewards[-10:]
                current_game_state = np.append(current_game_state[:, :, 1:], new_state.gray_image().reshape(21, 79, 1),
                                               axis=2)

                agent.replay_buffer.add_experience(act, new_state.gray_image()[0], rew, done)

                # Learning step
                if agent.replay_buffer.count > MIN_REPLAY_SIZE:
                    loss, _ = agent.learn(BATCH_SIZE, GAMMA, epsilon, PRIORITY_SCALE)
                    all_losses.append(loss)
                    all_losses = all_losses[-100:]
                    tf.summary.scalar('Loss', loss, step)

                if step % TARGET_UPDATE_FREQUENCY == 0 and step > MIN_REPLAY_SIZE:
                    agent.update_target_network()

                if step % 10 == 0:
                    tf.summary.scalar('Reward', np.mean(all_rewards), step)
                    tf.summary.scalar('Losses', np.mean(all_losses), step)

                if done:
                    dlvl = new_state.dungeon_level
                    env.reset()
                    all_rewards.append(episode_reward)
                    tf.summary.scalar('Evaluation score', episode_reward, episode)
                    tf.summary.scalar('Dungeon level', dlvl, episode)
                    print('Episode', episode, 'done.')
                    print('Reward this game', episode_reward)
                    print('Average reward current interval', np.mean(all_rewards))
                    print('Epsilon', epsilon, '\n')
                    episode_reward = 0
                    episode += 1

                if step % STEPS_PER_INTERVAL == 0 and step > 0:
                    print('\nInterval', intr)
                    agent.save(intr)
                    tf.summary.scalar('Interval score', np.mean(interval_rewards), intr)
                    interval_rewards = []
                    intr += 1

    except KeyboardInterrupt:
        print("Exiting~")
        writer.close()
    env.close()

# †昇天†
