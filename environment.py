from tensorflow.keras.layers import Dense, Flatten, InputLayer, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA

"""
Generate the communication distance matrix
"""
np.random.seed(0) # For replicable results
b = np.random.random_sample((10, 10))
distance_table = (b + b.T)/2
for i in range(10):
  distance_table[i][i] = 0


class Env:
    def __init__(self):
        self.client_num = 10
        self.OBSERVATION_SPACE_VALUES = (1, 100, 1)
        self.ACTION_SPACE_SIZE = self.client_num
        self.client_list = []
        self.data_list = []
        self.label_list = []
        self.acc_list = []
        self.img_shape = (28, 28, 1)
        self.pca_fl = PCA(n_components=10) # State vector size
        self.max_iteration_ep = 35

    def build_edge_discriminator(self):
        opt = Adam(learning_rate =0.001)

        model = Sequential()
        model.add(InputLayer(input_shape=self.img_shape))
        model.add(Conv2D(filters=20, kernel_size=5, strides=(1, 1), activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Conv2D(filters=50, kernel_size=5, strides=(1, 1), activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(units=10, activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        return model

    def reset(self, x_train, y_train, x_test, y_test):
        self.episode_step = 0
        self.gmodel = self.build_edge_discriminator()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.client_list = []
        self.data_list = []
        self.label_list = []
        self.acc_list.append([])
        for n in range(self.client_num):
          self.client_list.append(self.build_edge_discriminator())
          self.client_list[n].set_weights(self.gmodel.get_weights())

          main = np.isin(y_train, [n%10])
          all_label = list(range(10))
          all_label.pop(n%10)
          sub = np.isin(y_train, all_label)
          data_main, label_main = self.x_train[main][n//10*400:(n//10+1)*400], self.y_train[main][n//10*400:(n//10+1)*400]
          data_sub, label_sub = self.x_train[sub][n//10*100:(n//10+1)*100], self.y_train[sub][n//10*100:(n//10+1)*100]
          data_client, label_client = shuffle(np.concatenate((data_main, data_sub), axis = 0), to_categorical(np.concatenate((label_main, label_sub),axis = 0), 10))
          self.data_list.append(data_client)
          self.label_list.append(label_client)

        local_state = np.array(([np.concatenate( [w.flatten() for w in m.get_weights()] ) for m in self.client_list]))
        global_state = np.concatenate( [w.flatten() for w in self.gmodel.get_weights()]).reshape(1, -1)
        local_state = self.pca_fl.fit_transform(local_state).flatten()
        observation = local_state


        return observation

    def step(self, former_action, action):
        self.episode_step += 1
        local_update = []
        for c in action:
          self.client_list[c].fit(self.data_list[c], self.label_list[c], epochs=1, batch_size=32, verbose=0)
          local_update.append(self.client_list[c].get_weights())

        local_update = np.array(local_update, dtype=object)
        global_update = self.gmodel.get_weights()
        for w in range(len(self.client_list[0].get_weights())):
           global_update[w] = np.mean(local_update[:, w])


        self.gmodel.set_weights(global_update)

        local_state = np.array(([np.concatenate( [w.flatten() for w in m.get_weights()]) for m in self.client_list]))
        global_state = np.concatenate( [w.flatten() for w in self.gmodel.get_weights()]).reshape(1, -1)
        local_state = self.pca_fl.fit_transform(local_state).flatten()

        new_observation = local_state

        for local_model in self.client_list:
            local_model.set_weights(self.gmodel.get_weights())

        loss, acc = self.gmodel.evaluate(self.x_test,  to_categorical(self.y_test, 10), verbose=0)
        self.acc_list[-1].append(acc)

        goal = 0.8
        reward = 32**(acc - goal) - distance_table[former_action][action[0]]/10 - 1

        print("step: %s  acc: %s reward: %s " %(self.episode_step, acc, reward))

        done = False

        if acc >= goal or self.episode_step > self.max_iteration_ep:
           done = True

        return new_observation, reward, done