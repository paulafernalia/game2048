from keras.models import Sequential, clone_model, Model
from keras.optimizers import Adam
from keras.layers import Dropout, Dense, Lambda, Input
from keras import backend as K
import numpy as np


class DeepQ_MLP():

    def __init__(self, input_shape, epsilon=.05, learning_rate=0.01,
                 dropout=.1, discount_epsilon=1., discount_factor=1.,
                 c=100, beta_1=.9, warmup=50000, verbose=True):

        self.actions = ['D', 'U', 'L', 'R']
        self.num_actions = len(self.actions)
        self.gamma = discount_factor
        self.discount_epsilon = discount_epsilon
        self.c = c
        self.warmup = warmup
        self.verbose = verbose
        self.init_epsilon = epsilon
        self.construct_network(input_shape, dropout, learning_rate, beta_1)
        self.update_target_network()
        self.construct_trainable_model(learning_rate, beta_1)
        self.reset()

    def reset(self):
        """ Reset all counters and greedy parameter """
        self.c_count = 0
        self.a_count = -1
        self.epsilon = self.init_epsilon

    def random_action(self, action_list=None):
        """ Policy that selects one of the available actions at random """

        # sample from all actions
        if action_list is None:
            return np.random.choice(self.actions)

        # sample from a subset of actions
        else:
            return np.random.choice(action_list)

    def construct_network(self, input_shape, dropout, learning_rate, beta_1):
        """ Create multilayer perceptron """
        self.Qmodel = Sequential()
        self.Qmodel.add(Dropout(rate=dropout, input_shape=input_shape))
        self.Qmodel.add(Dense(128, activation='relu'))
        self.Qmodel.add(Dropout(rate=dropout))
        self.Qmodel.add(Dense(64, activation='relu'))
        self.Qmodel.add(Dense(self.num_actions, activation='linear'))

        opt = Adam(learning_rate=learning_rate, beta_1=beta_1)
        self.Qmodel.compile(loss='mse', optimizer=opt)

    def update_target_network(self):
        """ Clone structure and weights and compile """
        self.target_Qmodel = clone_model(self.Qmodel)
        self.target_Qmodel.set_weights(self.Qmodel.get_weights())

        # target network is never compiled
        self.target_Qmodel.compile(loss='mse', optimizer=Adam())

    def eps_greedy_action(self, phi, tabu):
        """Predict movement of game controler where is epsilon
        probability randomly move."""

        # increase counter of actions taken
        self.a_count += 1

        # if within the initial buffer before learning starts, random action
        aval_actions = None
        if self.a_count < self.warmup:

            if len(tabu) > 0:
                # Remove tabu actions from list of available actions
                aval_actions = [a for a in self.actions if a not in tabu]

            action = self.random_action(aval_actions)
            return action, None

        elif (self.a_count == self.warmup) and self.verbose:
            print('learning starts')

        # evaluate Q(phi, a) for each action
        qvalues = self.Qmodel.predict(phi, batch_size=1)[0]

        # generate random value
        randn = np.random.uniform()

        # eliminate tabu values from possible actions to pick
        aval_actions = None
        if len(tabu) > 0:
            if randn < self.epsilon:
                aval_actions = [a for a in self.actions if a not in tabu]
            else:
                # Update Qs to low values to ensure they are not picked
                tabu_idx = [i for i in range(self.num_actions) if self.actions[i] in tabu]
                qvalues[tabu_idx] = -9999

        # eps-greedy, select random action
        if randn < self.epsilon:
            action = self.random_action(aval_actions)
            a_i = self.action_str2idx(action)
        else:
            # select best action
            a_i = np.argmax(qvalues)
            action = self.actions[a_i]

        # update greedy parameter and action count
        self.epsilon *= self.discount_epsilon
        self.a_count += 1

        return action, qvalues[a_i]

    def get_target(self, batch):
        """ Calculate yj = rj + gamma argmaxQ or yj = rj (terminating state)
            This is the target value used to train the neural network and it
            uses the target network to make predictions
        """
        # initialise array to store yj values
        target = np.zeros((len(batch[0]), self.num_actions))

        # loop over samples in the minibatch
        for j in range(len(batch[0])):

            a0_i = self.action_str2idx(batch[1][j])
            r0 = batch[2][j]
            done = batch[3][j]
            s1 = batch[4][j]

            # if terminating state
            if done:
                target[j, a0_i] = r0
            else:
                qs_target = self.target_Qmodel.predict(s1)
                target[j, a0_i] = r0 + self.gamma * np.max(qs_target)

        return target

    def get_masks(self, actions):
        masks = np.zeros((len(actions), self.num_actions))

        for j in range(len(actions)):

            a0_i = self.action_str2idx(actions[j])
            masks[j, a0_i] = 1.

        return masks

    def one_step_gd(self, batch):
        """ Perform one step of gradient descent on (yj - Q(phi, aj, w))^2 """

        # get target values yj
        targets = self.get_target(batch)
        phi_input = np.vstack(batch[0])
        masks = self.get_masks(batch[1])
        dummy_targets = targets.max(axis=1)

        X = [phi_input, targets, masks]
        Y = [dummy_targets, targets]

        # update main network with one step of gradient descent
        # self.Qmodel.fit(X, Y, batch_size=len(X))
        metrics = self.train_model.train_on_batch(X, Y)

        # every fixed number of steps, update target network
        self.c_count += 1
        # print(self.c_count, self.c)

        if self.c_count == self.c:
            if self.verbose:
                print('* Target network updated')

            # update target network to be equal the main network
            self.update_target_network()

            # reset counter
            self.c_count = 0

    def action_str2idx(self, action):
        return np.argwhere(np.array(self.actions) == action)[0][0]

    def construct_trainable_model(self, learning_rate, beta_1):

        def masked_error(args):
            y_true, y_pred, mask = args
            loss = K.square(y_true - y_pred)
            loss *= mask
            return K.sum(loss, axis=-1)

        y_true = Input(name='y_true', shape=(self.num_actions,))
        mask = Input(name='mask', shape=(self.num_actions,))
        y_pred = self.Qmodel.output

        loss_out = Lambda(masked_error,
                          output_shape=(1, ),
                          name='loss')([y_true, y_pred, mask])

        train_model = Model([self.Qmodel.input] + [y_true, mask],
                            outputs=[loss_out])

        # loss is computed in Lambda layer
        losses = [
            lambda y_true, y_pred: y_pred,
            # lambda y_true, y_pred: K.zeros_like(y_pred)
        ]
        opt = Adam(learning_rate=learning_rate, beta_1=beta_1)
        train_model.compile(optimizer=opt, loss=losses)

        self.train_model = train_model
