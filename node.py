from tensorflow.keras.layers import Dense, Flatten, InputLayer, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import numpy as np

"""DQN Agent
This file includes the reinforcement learning algorithm for 
learning an optimized communication policy based on observations 
of model parameters and the correlated rewards.
"""

class DQNAgent:
    def __init__(self, ACTION_SPACE_SIZE):
        self.REPLAY_MEMORY_SIZE = 50000  
        self.model = self.create_model(ACTION_SPACE_SIZE)
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)
        self.MINIBATCH_SIZE = 128
        self.DISCOUNT = 0.9 # Discount for future rewards

    def create_model(self, ACTION_SPACE_SIZE):
        model = Sequential()
        model.add(InputLayer(input_shape=(1, 100)))
        model.add(Flatten())
        model.add(Dense(units=500, activation="relu"))
        model.add(Dense(units=200, activation="relu"))
        model.add(Dense(ACTION_SPACE_SIZE, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains network every episode
    def train(self):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.MINIBATCH_SIZE:
            return

        batch_size = self.MINIBATCH_SIZE
        # Get a minibatch of random samples from memory replay table
        np.random.shuffle(self.replay_memory)
        minibatch = random.sample(self.replay_memory, batch_size)
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array(([transition[0] for transition in minibatch]))
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        new_current_states = np.array(([transition[3] for transition in minibatch]))
        future_qs_list = self.model.predict(new_current_states)

        X = []
        y = []

        # Enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        y = self.normalize(y)
        # Fit on all samples as one batch
        self.model.fit(np.array((X)), np.array((y)), epochs = 1, batch_size=16, verbose=1)

    # Query for Q values given current observation
    def get_qs(self, state):
        return self.model.predict(state.reshape(1, 100))[0]

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
           return v
        return v / norm 
