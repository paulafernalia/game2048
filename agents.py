from keras.models import load_model, Sequential, clone_model
from keras.optimizers import Adam
from keras.layers import Dropout, Dense
import numpy as np

class DeepQ_MLP():
    
    def __init__(self, epsilon=.05, learning_rate=0.01, dropout=.1, discount_epsilon=1., 
                 discount_factor=1., c=100, beta_1=.9, warmup=50000, verbose=True):
        
        self.actions = ['D', 'U', 'L', 'R']
        self.learning_rate=learning_rate
        self.beta_1 = beta_1
        self.dropout = dropout
        self.gamma = discount_factor
        self.discount_epsilon = discount_epsilon
        self.construct_network()
        self.update_target_network()
        self.c = c
        self.warmup = warmup
        self.verbose = verbose
        self.init_epsilon = epsilon
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
    
    
    def construct_network(self):
        """ Create multilayer perceptron """
        
        self.Qmodel = Sequential()
        self.Qmodel.add(Dropout(rate=self.dropout, input_shape=(16*13,)))
        self.Qmodel.add(Dense(128, activation='relu'))
        self.Qmodel.add(Dropout(rate=self.dropout))
        self.Qmodel.add(Dense(64, activation='relu'))
        self.Qmodel.add(Dense(len(self.actions), activation='linear'))
        
        opt = Adam(learning_rate=self.learning_rate, beta_1=self.beta_1)
        self.Qmodel.compile(loss='mse', optimizer=opt)
        
        
    def update_target_network(self):
        """ Clone structure and weights and compile """
        self.target_Qmodel = clone_model(self.Qmodel)
        self.target_Qmodel.set_weights(self.Qmodel.get_weights())
        
        # target network is never compiled
        self.target_Qmodel.compile(loss='mse', optimizer=Adam())
    
    
    def qlearning_action(self, phi, tabu):
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
                print(tabu)
                print(aval_actions)
            else:
                # Update Q(phi,a) of tabu actions to a low value to ensure they are not picked
                tabu_idx = [i for i in range(len(self.actions)) if self.actions[i] in tabu]
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
        target = np.zeros(len(batch[0]))

        # loop over samples in the minibatch
        for j in range(len(batch[0])):

            # if terminating state
            if batch[3][j]:
                target[j] = batch[2][j]
            else:
                qmax_target = self.target_Qmodel.predict(batch[0][j])
                target[j] = batch[2][j] + self.gamma * np.argmax(qmax_target)   
                
        return target
             
        
    def perform_one_step_gd(self, batch):
        """ Perform one step of gradient descent on (yj - Q(phi, aj, w))^2 """
        
        # get target values yj
        Y = self.get_target(batch)
        X = np.vstack(batch[0])
        
        # update main network with one step of gradient descent
        self.Qmodel.fit(X, Y, batch_size=len(X))
        
        # every fixed number of steps, update target network
        self.c_count += 1
        
        if self.c_count == self.c:
            
            if self.verbose:
                print('target network updated')
                
            # update target network to be equal the main network
            self.update_target_network()
            
            # reset counter
            self.c_count = 0
            
            
    def action_str2idx(self, action):
        return np.argwhere(np.array(self.actions) == action)[0][0]
            
#     def save_model