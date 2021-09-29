class DQNAgent:
    def __init__(self):
        self.state_num = 1
        self.model = self.create_model()
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    def create_model(self):
        model = Sequential()
        model.add(InputLayer(input_shape=(self.state_num, 100)))
        model.add(Flatten())
        model.add(Dense(units=500, activation="relu"))
        model.add(Dense(units=200, activation="relu"))
        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains network every step during episode
    def train(self):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MINIBATCH_SIZE:
            return

        batch_size = MINIBATCH_SIZE
        # Get a minibatch of random samples from memory replay table
        np.random.shuffle(self.replay_memory)
        minibatch = random.sample(self.replay_memory, batch_size)
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        y = normalize(y)
        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), epochs = 1, batch_size=16, verbose=1)

    # Queries network for Q values given current observation
    def get_qs(self, state):
        return self.model.predict(state.reshape(1, 1, 100))[0]
