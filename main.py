import numpy as np
from tensorflow.keras import backend, losses
from collections import deque
import random
import pickle
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, InputLayer, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from tensorflow.keras.utils.np_utils import to_categorical
from node import * 
from environment import *

"""
Import MNIST
"""
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train.astype('float32')/255.0).reshape(-1,28,28,1)
x_test = (x_test.astype('float32')/255.0).reshape(-1,28,28,1)


"""
Generate the communication distance matrix
"""
np.random.seed(0) # For replicable results
b = np.random.random_sample((10, 10))
distance_table = (b + b.T)/2
for i in range(10):
  distance_table[i][i] = 0


"""
Hyperparameters of DQN agents
"""
EPISODES = 120
max_iteration_ep = 35
DISCOUNT = 0.9
REPLAY_MEMORY_SIZE = 50000
MINIBATCH_SIZE = 128
# Exploration settings
epsilon = 1  # Not a constant, going to be decayed
EPSILON_DECAY = 0.02
pca_fl = PCA(n_components=10) # State vector size


env = Env() # Restart experiments
ep_rewards = []
agent = DQNAgent()
reward_list = []
starter = 0 # Set the starter node


# Iterate over episodes
for episode in range(1, EPISODES + 1):
    print("episode:%s" %episode)
    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    # Reset environment and get initial state
    current_state = env.reset()
    # Reset flag and start iterating until episode ends
    done = False
    former_action = 0
    action_num = 1
    round = 0
    while not done:
        round =round+1
        action = []

        if np.random.random() > epsilon:
            # Get action from the DQN agent
            inference = agent.get_qs(current_state)

            for j in range(action_num):
              loc = list(inference).index(max(inference))
              action.append(loc)
              inference[loc] = min(inference) - 1
            print("================** %s **=================" %action)
        else:
            j = action_num
            while j > 0 :
              # Get random action
              temp_action = np.random.randint(0, env.ACTION_SPACE_SIZE)
              if temp_action not in action:
                  action.append(temp_action)
                  j = j - 1
            print("================ %s =================" %action)

        if round == 1: 
          action = [starter]
          print("================ %s =================" %action)


        new_state, reward, done = env.step(former_action, action)
        former_action = action[0]

        # Every step we update replay memory and train the DQN model
        if round >1:
          agent.update_replay_memory((current_state, action, reward, new_state, done))

        # Transform new state and count reward
        current_state = new_state
        episode_reward += reward

        if done:
            print('Total training rewards: {} after n steps = {} with final reward = {}'.format(episode_reward, episode, reward))
            reward_list.append(episode_reward)
            # Save the training results for every five episodes
            if episode % 5 == 0:
                np.save('reward_100.npy', np.array((reward_list)), allow_pickle=True)
                np.save('acc_100.npy', env.acc_list, allow_pickle = True)

    # Decay epsilon
    agent.train()
    epsilon = epsilon * np.exp(-EPSILON_DECAY)
