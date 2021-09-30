import numpy as np
import pickle
from tensorflow.keras.datasets import mnist
from node import * 
from environment import *

"""
Import MNIST
"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = (x_train.astype('float32')/255.0).reshape(-1,28,28,1)
x_test = (x_test.astype('float32')/255.0).reshape(-1,28,28,1)


"""
Hyperparameters of DQN agents
"""
EPISODES = 120
DISCOUNT = 0.9
MINIBATCH_SIZE = 128
# Exploration settings
epsilon = 1  # Not a constant, going to be decayed
EPSILON_DECAY = 0.02

# Restart experiments
env = Env() 
agent = DQNAgent(env.ACTION_SPACE_SIZE)
ep_rewards = []
reward_list = []
starter = 0 # Set the starter node


# Iterate over episodes
for episode in range(1, EPISODES + 1):
    print("episode:%s" %episode)
    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    # Reset environment and get initial state
    current_state = env.reset(x_train, y_train, x_test, y_test)
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
                np.save('reward.npy', np.array((reward_list)), allow_pickle=True)
                np.save('acc.npy', env.acc_list, allow_pickle = True)

    # Decay epsilon
    agent.train()
    epsilon = epsilon * np.exp(-EPSILON_DECAY)
