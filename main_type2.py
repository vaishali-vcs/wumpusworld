from random import randint, choice
import numpy as np
from collections import deque
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import tensorflow as tf
from statistics import mean
# use tf2
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tqdm import tqdm
import os, time, sys, random
import pandas as pd

sys.path.append(os.getcwd())

from wumpusworld.envs.WumpusGym import WumpusWorldEnv
from BeelineAgent import Agent, Action
# For more repetitive results
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
PATH = ""

# Create models folder
if not os.path.isdir(f'{PATH}models'):
    os.makedirs(f'{PATH}models')
# Create results folder
if not os.path.isdir(f'{PATH}results'):
    os.makedirs(f'{PATH}results')

WORLD_SIZE= 4
Inputshape = 72

class ModifiedTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, name)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

#######################################################################################################

enc_orientation = [[1, 0, 0, 0],   #Right
                   [0, 1, 0, 0],    #Up
                   [0, 0, 1, 0],    # Left
                   [0, 0, 0, 1]]    #Down
enc_heardscream = [0, 1]
enc_glitter = [0, 1]
enc_hasgold = [0, 1]
enc_hasarrow = [0, 1]


# Agent class
class DQNAgent:
    def __init__(self, name, env, dense_list, util_list):
        self.action_space = env.action_space
        self.dense_list = dense_list
        self.name = [str(name) + " | " + "".join(
            str(d) + "D | " for d in dense_list) + "".join(u + " | " for u in util_list)][0]

        # Main model
        self.model = self.create_model( self.dense_list)

        # Target network
        self.target_model = self.create_model(self.dense_list)
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(name, log_dir="{}logs/{}-{}".format(PATH, name, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        self.visitedlocations = []
        self.breezelocations = []
        self.stenchlocations = []
        self.heardscream = False
        self.hasarrow = True
        self.hasgold = True
        self.glitter = False

    # Creates the model with the given specifications:
    def create_model(self, dense_list):
        # Defines the input layer with shape = ENVIRONMENT_SHAPE
        model = tf.keras.Sequential()
        model.add(Input((1,Inputshape, )))

        # Creating the dense layers:
        for d in dense_list:
            model.add(Dense(units=d, activation='relu'))
        # The output layer has 5 nodes (one node per action)
        model.add(Dense(units=len(self.action_space), activation='linear', name='output'))

        model.compile(optimizer=Adam(lr=0.001),
                      loss={'output': 'mse'},
                      metrics={'output': 'accuracy'})

        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[0, action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(x=np.array(X),
                       y=np.array(y),
                       batch_size=MINIBATCH_SIZE, verbose=0,
                       shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(state)

    def processPercepts(self, WORLD_SIZE, percept):
        """ BeeLineAgent_Process: called with new percepts after each action to return the next action """

        percept_str = ""
        if percept['stench']:
            self.stenchlocations.append(str(percept['x']) + ',' + str(percept['y'])) if str(percept['x']) + ',' + str(percept['y']) not in self.stenchlocations else None

        if percept['breeze']:
            self.breezelocations.append(str(percept['x']) + ',' + str(percept['y'])) if str(percept['x']) + ',' + str(percept['y']) not in self.breezelocations else None
        if percept['glitter']:
            self.glitter = True
        if percept['scream']:
           self.heardscream = True
        if percept['arrow']:
            self.hasarrow = True
        if percept['gold']:
            self.hasgold = True

        self.visitedlocations.append(str(percept['x']) + ',' + str(percept['y'])) if str(percept['x']) + ',' + str(
            percept['y']) not in self.visitedlocations else None

        np_agentlocs = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=np.uint8)
        np_breezelocs = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=np.uint8)
        np_stenchlocs = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=np.uint8)
        np_visitedlocs = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=np.uint8)

        np_agentlocs[percept['x'], percept['y']] = 1

        for loc in self.breezelocations:
            np_breezelocs[int(loc.split(',')[0]), int(loc.split(',')[1])] = 1

        for loc in self.stenchlocations:
            np_stenchlocs[int(loc.split(',')[0]), int(loc.split(',')[1])] = 1

        for loc in self.visitedlocations:
            np_visitedlocs[int(loc.split(',')[0]), int(loc.split(',')[1])] = 1

        np_breezelocs = np_breezelocs + np.random.rand(np_breezelocs.shape[0], np_breezelocs.shape[1]) / 10
        np_stenchlocs = np_stenchlocs + np.random.rand(np_stenchlocs.shape[0], np_stenchlocs.shape[1]) / 10
        np_visitedlocs = np_visitedlocs + np.random.rand(np_visitedlocs.shape[0], np_visitedlocs.shape[1]) / 10
        np_agentlocs = np_agentlocs + np.random.rand(np_agentlocs.shape[0], np_agentlocs.shape[1]) / 10

        orient = enc_orientation[percept['direction'].value]
        heardscream = enc_heardscream[self.heardscream]
        glitter = enc_glitter[self.glitter]
        hasgold = enc_hasgold[self.hasgold]
        hasarrow = enc_hasarrow[self.hasarrow]

        state_p1 = np.stack([np_agentlocs, np_stenchlocs, np_stenchlocs, np_visitedlocs]).flatten()
        state_p2 = np.append(orient, np.array([heardscream, glitter, hasgold, hasarrow]))
        state = np.append(state_p1, state_p2)

        return np.array([state])

######################################################################################


def save_model_and_weights(agent, model_name, episode, max_reward, average_reward, min_reward):
    checkpoint_name = f"{model_name}| Eps({episode}) | max({max_reward:_>7.2f}) | avg({average_reward:_>7.2f}) | min({min_reward:_>7.2f}).model"
    agent.model.save(f'{PATH}models/{checkpoint_name}')
    best_weights = agent.model.get_weights()
    return best_weights
######################################################################################
# ## Constants:
# RL Constants:
DISCOUNT               = 0.99
REPLAY_MEMORY_SIZE     = 1_000   # How many last steps to keep for model training
# MIN_REPLAY_MEMORY_SIZE = 1_000   # Minimum number of steps in a memory to start training
UPDATE_TARGET_EVERY    = 20      # Terminal states (end of episodes)
MIN_REWARD             = 1000    # For model save
SAVE_MODEL_EVERY       = 1000    # Episodes
SHOW_EVERY             = 20      # Episodes
EPISODES               = 1000  # Number of episodes
#  Stats settings
AGGREGATE_STATS_EVERY = 20  # episodes
SHOW_PREVIEW          = False
######################################################################################
# Models Arch :
 # [{ [dense_list], [util_list], MINIBATCH_SIZE, {EF_Settings}, {ECC_Settings}} ]

models_arch = [{ "dense_list": [150, 100], "util_list": ["ECC2", "1A-5Ac"],
                "MINIBATCH_SIZE": 128, "best_only": False,
                "EF_Settings": {"EF_Enabled": False}, "ECC_Settings": {"ECC_Enabled": False}},

               { "dense_list": [128, 100, 64], "util_list": ["ECC2", "1A-5Ac"],
                "MINIBATCH_SIZE": 128, "best_only": False,
                "EF_Settings": {"EF_Enabled": False}, "ECC_Settings": {"ECC_Enabled": False}},

               { "dense_list": [64, 32], "util_list": ["ECC2", "1A-5Ac"],
                "MINIBATCH_SIZE": 128, "best_only": False,
                "EF_Settings": {"EF_Enabled": True, "FLUCTUATIONS": 2},
                "ECC_Settings": {"ECC_Enabled": True, "MAX_EPS_NO_INC": int(EPISODES * 0.2)}}]

# A dataframe used to store grid search results
res = pd.DataFrame(columns=["Model Name", "Dense Layers", "Batch Size", "ECC", "EF",
                            "Best Only", "Average Reward", "Best Average", "Epsilon 4 Best Average",
                            "Best Average On", "Max Reward", "Epsilon 4 Max Reward", "Max Reward On"
                            ])


######################################################################################


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


def search():
    global MINIBATCH_SIZE, env, agent, agent, res
    games_won ={}

    for i, m in enumerate(models_arch):
        games_won[i] = 0
        MINIBATCH_SIZE = m["MINIBATCH_SIZE"]

        # Exploration settings :
        # Epsilon Fluctuation (EF):
        EF_Enabled = m["EF_Settings"]["EF_Enabled"]  # Enable Epsilon Fluctuation
        MAX_EPSILON = 1  # Maximum epsilon value
        MIN_EPSILON = 0.001  # Minimum epsilon value
        if EF_Enabled:
            FLUCTUATIONS = m["EF_Settings"]["FLUCTUATIONS"]  # How many times epsilon will fluctuate
            FLUCTUATE_EVERY = int(EPISODES / FLUCTUATIONS)  # Episodes
            EPSILON_DECAY = MAX_EPSILON - (MAX_EPSILON / FLUCTUATE_EVERY)
            epsilon = 1  # not a constant, going to be decayed
        else:
            EPSILON_DECAY = MAX_EPSILON - (MAX_EPSILON / (0.8 * EPISODES))
            epsilon = 1  # not a constant, going to be decayed

        # Initialize some variables:
        best_average = -100
        best_score = -100

        # Epsilon Conditional Constantation (ECC):
        ECC_Enabled = m["ECC_Settings"]["ECC_Enabled"]
        avg_reward_info = [
            [1, best_average, epsilon]]  # [[episode1, reward1 , epsilon1] ... [episode_n, reward_n , epsilon_n]]
        max_reward_info = [[1, best_score, epsilon]]
        if ECC_Enabled: MAX_EPS_NO_INC = m["ECC_Settings"][
            "MAX_EPS_NO_INC"]  # Maximum number of episodes without any increment in reward average
        eps_no_inc_counter = 0  # Counts episodes with no increment in reward

        # For stats
        ep_rewards = [best_average]
        env = WumpusWorldEnv()
        agent = DQNAgent(f"M{i}", env, m["dense_list"], m["util_list"])

        MODEL_NAME = agent.name
        best_weights = [agent.model.get_weights()]

        # Iterate over episodes
        for episode in tqdm(range(1, EPISODES + 1), ascii=True ,unit='episodes'):
            # print("episode {} / {}".format(episode, EPISODES))
            if m["best_only"]: agent.model.set_weights(best_weights[0])
            # agent.target_model.set_weights(best_weights[0])

            score_increased = False
            # Update tensorboard step every episode
            agent.tensorboard.step = episode

            # Restarting episode - reset episode reward and step number
            episode_reward = 0
            step = 1
            action = 0

            game_over = False
            done = False

            while not done:
                # Reset environment and get initial state
                percept = env.reset()
                current_state = agent.processPercepts(WORLD_SIZE, percept)

                # This part stays mostly the same, the change is to query a model for Q values
                if np.random.random() > epsilon:
                    # Get action from Q table
                    action = np.argmax(agent.get_qs(current_state))

                else:
                    # Get random action
                    action = choice(env.action_space)

                percept, reward, game_over = env.step(action)
                new_state = agent.processPercepts(WORLD_SIZE, percept)
                reward = int(reward)
                # Transform new continuous state to new discrete state and count reward
                episode_reward += reward

                agent.update_replay_memory((current_state, action, reward, new_state, game_over))
                agent.train(game_over, step)

                current_state = new_state
                step += 1
                done = game_over

            if episode_reward >= 1000:
                games_won[i]+=1

            if ECC_Enabled: eps_no_inc_counter += 1
            # Append episode reward to a list and log stats (every given number of episodes)
            ep_rewards.append(episode_reward)

            if not episode % AGGREGATE_STATS_EVERY:
                average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
                agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                               epsilon=epsilon)

                # Save models, but only when avg reward is greater or equal a set value
                if not episode % SAVE_MODEL_EVERY:
                    # Save Agent :
                    _ = save_model_and_weights(agent, MODEL_NAME, episode, max_reward, average_reward, min_reward)

                if average_reward > best_average:
                    best_average = average_reward
                    # update ECC variables:
                    avg_reward_info.append([episode, best_average, epsilon])
                    eps_no_inc_counter = 0
                    # Save Agent :
                    best_weights[0] = save_model_and_weights(agent, MODEL_NAME, episode, max_reward, average_reward,
                                                             min_reward)

                if ECC_Enabled and eps_no_inc_counter >= MAX_EPS_NO_INC:
                    epsilon = avg_reward_info[-1][2]  # Get epsilon value of the last best reward
                    eps_no_inc_counter = 0

            if episode_reward > best_score:
                try:
                    best_score = episode_reward
                    max_reward_info.append([episode, best_score, epsilon])

                    # Save Agent :
                    best_weights[0] = save_model_and_weights(agent, MODEL_NAME, episode, max_reward, average_reward,
                                                             min_reward)

                except:
                    pass

            # Decay epsilon
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

            # Epsilon Fluctuation:
            if EF_Enabled:
                if not episode % FLUCTUATE_EVERY:
                    epsilon = MAX_EPSILON

        # Get Average reward:
        average_reward = round(sum(ep_rewards) / len(ep_rewards), 2)

        # Update Results DataFrames:
        res = res.append(
            {"Model Name": MODEL_NAME,"Dense Layers": m["dense_list"],
             "Batch Size": m["MINIBATCH_SIZE"], "ECC": m["ECC_Settings"], "EF": m["EF_Settings"],
             "Best Only": m["best_only"], "Average Reward": average_reward,
             "Best Average": avg_reward_info[-1][1], "Epsilon 4 Best Average": avg_reward_info[-1][2],
             "Best Average On": avg_reward_info[-1][0], "Max Reward": max_reward_info[-1][1],
             "Epsilon 4 Max Reward": max_reward_info[-1][2], "Max Reward On": max_reward_info[-1][0]}
            , ignore_index=True)
        res = res.sort_values(by='Best Average')
        avg_df = pd.DataFrame(data=avg_reward_info, columns=["Episode", "Average Reward", "Epsilon"])
        max_df = pd.DataFrame(data=max_reward_info, columns=["Episode", "Max Reward", "Epsilon"])

        # Save dataFrames
        res.to_csv(f"{PATH}results/Results.csv")
        avg_df.to_csv(f"{PATH}results/{MODEL_NAME}-Results-Avg.csv")
        max_df.to_csv(f"{PATH}results/{MODEL_NAME}-Results-Max.csv")

    return games_won


# Grid Search:


if __name__ == '__main__':
    games_won = search()
