import os, sys, time, random
import numpy as np
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import datetime
from statistics import mean
from tqdm import tqdm
import pandas as pd

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
# use tf2
# https://github.com/VXU1230/Medium-Tutorials/blob/master/dqn/cart_pole.py
# https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998

sys.path.append(os.getcwd())

from wumpusworld.envs.WumpusGym import WumpusWorldEnv
from BeelineAgent import Agent, Action

WORLD_SIZE = 4


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

    def train(self, TargetNet, isdone):
        if isdone:
            pass
        elif len(self.experience['s']) < self.min_experiences:
            return 0

        # ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        ids = range(0, len(self.experience['s']))
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
        # if isinstance(loss, int):
        #     print(loss)
        # else:
        #     print(loss.numpy())

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


def play_game(agent, env, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    iter = 0
    done = False
    observations = env.reset()
    losses = list()
    gameswon = 0
    steps = 0
    while not done:
        state = agent.processPercepts(WORLD_SIZE, observations)
        # print(state[0:16])
        # print(state[16:32])
        # print(state[32:48])
        # print(state[48:64])
        # print(state[64:])
        action = TrainNet.get_action(state, epsilon)
        steps += 1
        # print(action_to_string(action))
        prev_observations = state
        observations, reward, done = env.step(action)

        state2 = agent.processPercepts(WORLD_SIZE, observations)

        rewards += int(reward)
        if done:
            env.reset()

        exp = {'s': prev_observations, 'a': action, 'r': int(reward), 's2': state2, 'done': done}
        TrainNet.add_experience(exp)

    loss = TrainNet.train(TargetNet, done)
    if isinstance(loss, int):
        losses.append(loss)
    else:
        losses.append(loss.numpy())
    iter += 1
    if iter % copy_step == 0:
        TargetNet.copy_weights(TrainNet)

    if int(reward) > 0:
        gameswon += 1

    # print("steps=", steps)
    return rewards, mean(losses), gameswon, steps


def playgame():
    global res
    sumgameswon = {}
    for i, m in enumerate(models_arch):

        env = WumpusWorldEnv()
        # env.render()
        agent = Agent()
        gamma = 0.99
        copy_step = 25
        num_states = l1
        num_actions = len(env.action_space)
        hidden_units = m["dense_list"]
        max_experiences = m["MINIBATCH_SIZE"]
        min_experiences = m["MINIBATCH_SIZE"]
        batch_size = m["MINIBATCH_SIZE"]
        lr = 1e-2
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = m["modelname"]
        log_dir = 'logs/dqn/' + model_name + '/' + current_time
        summary_writer = tf.summary.create_file_writer(log_dir)

        TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
        TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
        N = 5000
        total_rewards = np.empty(N)
        epsilon = 0.99
        decay = 0.9999
        min_epsilon = 0.1
        sumgameswon[i] = 0
        recordwintime = True
        total_time_sec = 0
        total_time_min = 0

        for n in tqdm(range(N)):
            startTime = time.time()  # Used to count episode training time
            epsilon = max(min_epsilon, epsilon * decay)
            total_reward, losses, gameswon, stepstaken = play_game(agent, env, TrainNet, TargetNet, epsilon, copy_step)
            sumgameswon[i] += gameswon
            total_rewards[n] = total_reward
            avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
            with summary_writer.as_default():
                tf.summary.scalar('episode reward', total_reward, step=n)
                tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)
                tf.summary.scalar('average loss)', losses, step=n)
                tf.summary.scalar('stepstaken)', stepstaken, step=n)
                tf.summary.scalar('gameswon)', gameswon, step=n)

            if gameswon == 1 and recordwintime:
                recordwintime = False
                endTime = time.time()
                total_time_sec = round((endTime - startTime))
                total_time_min = round((endTime - startTime) / 60, 2)

        res = res.append(
            {"Model Name": model_name, "Dense Layers": m["dense_list"],
             "Batch Size": m["MINIBATCH_SIZE"], "Total games won": sumgameswon[i],
             "time for 1st win(min)": total_time_min, "time for 1st win(sec)": total_time_sec}
            , ignore_index=True)

        # print("gameswon= by model = ", sumgameswon[i], model_name)


models_arch = [{"modelname": 'M150_100', "dense_list": [150, 100], "MINIBATCH_SIZE": 32},
               {"modelname": 'M150_100', "dense_list": [150, 100], "MINIBATCH_SIZE": 64},
               {"modelname": 'M150_100', "dense_list": [150, 100], "MINIBATCH_SIZE": 128},

               {"modelname": 'M150_100', "dense_list": [200, 200], "MINIBATCH_SIZE": 32},
               {"modelname": 'M150_100', "dense_list": [200, 200], "MINIBATCH_SIZE": 64},
               {"modelname": 'M150_100', "dense_list": [200, 200], "MINIBATCH_SIZE": 128},

               {"modelname": 'M128_100_64', "dense_list": [128, 100, 64], "MINIBATCH_SIZE": 32},
               {"modelname": 'M128_100_64', "dense_list": [128, 100, 64], "MINIBATCH_SIZE": 64},
               {"modelname": 'M128_100_64', "dense_list": [128, 100, 64], "MINIBATCH_SIZE": 128},

               {"modelname": 'M64_32', "dense_list": [64, 32], "MINIBATCH_SIZE": 32},
               {"modelname": 'M64_32', "dense_list": [64, 32], "MINIBATCH_SIZE": 64},
               {"modelname": 'M64_32', "dense_list": [64, 32], "MINIBATCH_SIZE": 128}]

# A dataframe used to store grid search results
res = pd.DataFrame(columns=["Model Name", "Dense Layers", "Batch Size",
                            "Total games won", "time for 1st win(min)", "time for 1st win(sec)"
                            ])

if __name__ == '__main__':
    # Grid Search:
    playgame()
