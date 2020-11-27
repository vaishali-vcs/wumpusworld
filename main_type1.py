import sys
import os
import random
import numpy as np
from collections import deque
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import tensorflow as tf
import datetime
from statistics import mean
# use tf2
# https://github.com/VXU1230/Medium-Tutorials/blob/master/dqn/cart_pole.py
# https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998

sys.path.append(os.getcwd())

from wumpusworld.envs.WumpusGym import WumpusWorldEnv
from BeelineAgent import Agent, Action

WORLD_SIZE= 4


class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = Adam(lr)
        self.gamma = gamma
        self.model = MyModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)
        i = 1

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=-1)
        actual_values = np.where(dones, rewards, rewards + self.gamma * value_next.squeeze())

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=-1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss


l1 = 72
l2 = 150
l3 = 100
l4 = 6

loss_fn = MSE
learning_rate = 1e-3
optmizer = None


def action_to_string(action):
    """ action_to_string: return a string from the given action """
    if action == Action.WALK:
        return "WALK"
    if action == Action.TURNRIGHT:
        return "TURNRIGHT"
    if action == Action.TURNLEFT:
        return "TURNLEFT"
    if action == Action.SHOOT:
        return "SHOOT"
    if action == Action.GRAB:
        return "GRAB"
    if action == Action.CLIMB:
        return "CLIMB"
    return "UNKNOWN ACTION"

#
# def buildnetwork():
#     m = Sequential(
#         [
#             Input(shape=(l1,), name="layer1"),
#             Dense(l2, activation="relu", name="layer2"),
#             Dense(l3, activation="relu", name="layer3"),
#             Dense(l4, name="layer4"),
#         ]
#     )
#     m.compile(optimizer=Adam(learning_rate=learning_rate), loss=MSE)
#
#     return m


# def getoptimizer():
#     return Adam(learning_rate=learning_rate)
#
#
# def getQmodels():
#     qmodel = buildnetwork()
#     # targetmodel.set_weights(qmodel.get_weights())
#     targetmodel = clone_model(qmodel)
#     targetmodel.set_weights(qmodel.get_weights())
#     opt = getoptimizer()
#     return qmodel, targetmodel, opt


def play_game(agent, env, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    iter = 0
    done = False
    observations = env.reset()
    losses = list()
    while not done:
        state = agent.processPercepts(WORLD_SIZE, observations)
        action = TrainNet.get_action(state, epsilon)
        prev_observations = state
        observations, reward, done = env.step(action)
        state2 = agent.processPercepts(WORLD_SIZE, observations)

        rewards += int(reward)
        if done:
            reward = -200
            env.reset()

        exp = {'s': prev_observations, 'a': action, 'r': int(reward), 's2': state2, 'done': done}
        TrainNet.add_experience(exp)
        loss = TrainNet.train(TargetNet)
        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
    return rewards, mean(losses)


def test_model(model):
    i = 0
    test_game = WumpusWorldEnv()
    agent = Agent()
    observations =env.reset()

    status = 1
    while status == 1:  # A
        state = agent.processPercepts(WORLD_SIZE, observations)
        action = model.get_action(state, epsilon)

        observations, reward, done = test_game.step(action)
        state2 = agent.processPercepts(WORLD_SIZE, observations)
        reward = int(reward)

        if reward != -1:
            if reward > 0:
                status = 2
            else:
                status = 0
        i += 1
        if i > 15:
            break

    win = True if status == 2 else False
    return win


def test(tnet):
    max_games = 1000
    wins = 0
    for i in range(max_games):
        win = test_model(tnet)
        if win:
            wins += 1
    win_perc = float(wins) / float(max_games)
    print("Games played: {0}, # of wins: {1}".format(max_games, wins))
    print("Win percentage: {}%".format(100.0 * win_perc))


if __name__ == '__main__':
    env = WumpusWorldEnv()
    agent = Agent()
    gamma = 0.99
    copy_step = 25
    num_states = l1
    num_actions = len(env.action_space)
    hidden_units = [200, 200]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32
    lr = 1e-2
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    N = 100#10000
    total_rewards = np.empty(N)
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.1

    for n in range(N):
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward, losses = play_game(agent, env, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
        with summary_writer.as_default():
            tf.summary.scalar('episode reward', total_reward, step=n)
            tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)
            tf.summary.scalar('average loss)', losses, step=n)
        if n % 100 == 0:
            print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards,
                  "episode loss: ", losses)
    print("avg reward for last 100 episodes:", avg_rewards)

    test(TrainNet)
