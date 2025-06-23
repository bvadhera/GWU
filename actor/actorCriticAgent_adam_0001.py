# Actor Critic and Agent code

 
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
import time
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
import csv
from csv import DictReader
import os
import random
import math
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input,Dense,Lambda,Concatenate 
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.layers import LeakyReLU

# Implement ReplayBuffer  to store the states , actions, rewards, new states 
# & terminal flags. Also the agent encounters and its adventures. We use numpy to 
# implement it as it is simpler and cleaner from implementation purposes.
# In short we use replay buffer to store experiences.
class RBuffer():
  # maxsize - max size of memory to bound it
  # statedim - input shape from our environment
  # naction - number of actions for our action space (contineous action space) 
  #           - it means that number of components to the action
  def __init__(self, maxsize, statedim, naction):

    # We need a memory counter starting with 0
    self.cnt = 0
    # mem cannot be unbounded and as we exceed memory size we will overwrite
    #  our earliest memory with the new one.
    self.maxsize = maxsize  
    # np.zeros() function  get a new array of given shape and type, filled with zeros.
    # we define  our state memory which is memory size by imput shape (statedim)
    self.state_memory = np.zeros((maxsize, *statedim), dtype=np.float32)
    # we define new state memory which is memory size by imput shape (statedim)
    self.next_state_memory = np.zeros((maxsize, *statedim), dtype=np.float32)
    # we will need an action memory which is memory size by number of actions
    self.action_memory = np.zeros((maxsize, naction), dtype=np.float32)
    # we will need an reward memory which is just shape memory size. It will be 
    # just an array of floating point numbers.
    self.reward_memory = np.zeros((maxsize,), dtype=np.float32)
    # just an array of floating point numbers.
    self.accuracy_memory = np.zeros((maxsize,), dtype=np.float32)
    #  We also need a terminal memory which will be of type boolean 
    self.done_memory = np.zeros((maxsize,), dtype= np.bool)

  # We need a function to store transitions where transition is 
  #   state,action,reward, new_state(Next_state) and terminal flag (done)
  def storexp(self, state, next_state, action, done, reward, accuracy):
    # First we need to know the position of the first available memory by doing modules of 
    # current memory counter and memory size
    index = self.cnt % self.maxsize   # Huber ?
    # Now we have the index we can go ahead saving our transitions 
    # (state, next_state, action, done, reward)
    self.state_memory[index] = state
    self.action_memory[index] = action
    self.reward_memory[index] = reward
    self.next_state_memory[index] = next_state
    # terminal memory - The value of terminal state is 0 as no future rewards 
    # follow from that terminal state therefore  we have to set the episode 
    # back to the initial state
    self.done_memory[index] = 1- int(done)
    self.accuracy_memory[index] = accuracy
    # Increment mem counter by 1
    self.cnt += 1

  # We need a function to  sample our buffer
  def sample(self, batch_size):
    # We want to know how much of the memory we have filled up
    max_mem = min(self.cnt, self.maxsize)
    # Now we take batch of numbers,  replace= False as
    #  once the memory is sampled from that range it will not be sampled again.
    # it prevents you from double sampling again same memory
    batch = np.random.choice(max_mem, batch_size, replace= False)  
    # Now we go ahead and de-reference our numpy arrays and return those at the end.
    states = self.state_memory[batch]
    next_states = self.next_state_memory[batch]
    rewards = self.reward_memory[batch]
    actions = self.action_memory[batch]
    dones = self.done_memory[batch]
    accuracy = self.accuracy_memory[batch]
    return states, next_states, rewards, actions, dones,accuracy



# Actor  network class
class Actor(tf.keras.Model):
    def __init__(self):
        super(Actor, self).__init__()    
        self.h1 = Dense(10, activation=LeakyReLU(alpha=0.1),  name='h1')
        self.h2 = Dense(10, activation=LeakyReLU(alpha=0.1),  name='h2')
        self.actor_output = Dense(2, activation='tanh', name='actor_output')

    #  We have call function for Actor for forward propagation operation.
    def call(self, predict_layer):
        x = self.h1(predict_layer)
        x = self.h2(x)
        x = self.actor_output(x)
        #print(sess.run(x))
     
        return x
            
    #  Freeze the Layers
    def freezeLayers(self):
        self.h1.trainable = False
        self.actor_output.trainable = False   

    #  Freeze the Layers
    def UnfreezeLayers(self):
        self.h1.trainable = True
        self.actor_output.trainable = True

# Critic network class
class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()    
        self.state_h1 = Dense(24, activation=LeakyReLU(alpha=0.1), name='state_h1')
        self.state_h2 = Dense(24, activation=LeakyReLU(alpha=0.1), name='state_h2')

        self.critic_output = Dense(1, activation='tanh', name='critic_output')

    #  We have call function for Critic for forward propagation operation.
    def call(self, predict_layer, action):
        x = self.state_h1(tf.concat([predict_layer, action], axis=1))
        x = self.state_h2(x)
        x = self.critic_output(x)
        return x
    
    def mid_call(self, predict_layer, action):
        x = self.state_h1(tf.concat([predict_layer, action], axis=1))
        x = self.state_h2(x)
        return x
     
    def early_call(self, predict_layer, action):
        x = self.state_h1(tf.concat([predict_layer, action], axis=1))
        return x
           
    #  Freeze the Layers
    def freezeLayers(self, predict_layer):
        self.state_h1.trainable = False
        self.critic_output.trainable = False   

    #  UnFreeze the Layers
    def unFreezeLayers(self, predict_layer):
        self.state_h1.trainable = True
        self.critic_output.trainable = True



# Agent network class
# We will need our environment for the max/min actions as 
# we will be adding noise to the output of our deep NN for some exploration.
class Agent():
  # Huber - what is action space here?
  def __init__(self):
    # instantiate out actual networks actor and crtic and target actor and target critic
    # In DDPG, we have target networks for both actor and critic 
    self.actor_main = Actor()
    self.actor_target = Actor()
    self.critic_main = Critic()
    self.critic_main2 = Critic()
    self.critic_target = Critic()
    self.critic_target2 = Critic()
    # batch size for our memory sampling
    self.batch_size = 300
    # number of our actions
    self.n_actions = 2  # As actor gives two dimension x,y output.
    # Adam optimizer for actor  
    self.a_opt = tf.keras.optimizers.Adam(0.001)
    # self.actor_target = tf.keras.optimizers.Adam(.001)
    self.c_opt1 = tf.keras.optimizers.Adam(0.0001)
    self.c_opt2 = tf.keras.optimizers.Adam(0.0001)
    
    # We dont need to be doing any gradient decent on 
    #    both actor and critic target networks. 
    #    We will be doing only soft network update on these target networks.
    #    In learning fucntion we dont call an update for the loss function for these.

    # self.actor_target = tf.keras.optimizers.Adam(.001)
    # self.critic_target = tf.keras.optimizers.Adam(.002)


    # maxsize for ReplyBuffer is defaulted to 1,000,000 # a million
    # input dimensions is env.observation_space.shape and actions is env.action_space.high
    #  env.observation_space.shape is state of the network Adam
    observation_space_shape = (12,)  # Ask Huber - assuming we have only two dimensional space
    #HACK
    self.memory = RBuffer(3010001,  observation_space_shape, 2)
    self.trainstep = 0
    self.replace = 5
    # We need gamma - a discount factor for update equation
    self.gamma =0.999

    # max/min actions for our environment 
    # Huber What are min/max (-1/+1) ?
    # Ask Huber - what are they in our case
    self.min_action = -1.0
    self.max_action = 1.0

    self.actor_update_steps = 2
    self.warmup = 2250000
    # default value for our soft update tau
    self.tau = 0.005
    # Note that we have compiled our target networks as we don’t want
    #  to get an error while copying weights from main networks to target networks.
    self.actor_target.compile(optimizer=self.a_opt)
    self.critic_target.compile(optimizer=self.c_opt1)
    self.critic_target2.compile(optimizer=self.c_opt2)

  # Choose an Action. It will take current state of environment as
  #  input as well as evaluate=False to train vs test. 
  #  Just test agent without adding the noise to get pure deterministic output.
  def act(self, state, evaluate=False):
    
      if self.trainstep > self.warmup:
            evaluate = True
    # For action selection, first, we convert our state into a tensor and then pass it
    #  to the actor-network.
    # Therefore we convert the state to tensor & add extra dimension to our 
    # observation(state) the batch dimension that is what deep NN expect as input.
    # They expect the batch dimension.
    #  np.asarray 
      #print ("state type", type(state))
      # Huber Replace to avoid Argument must be a dense tensor:got shape [1, 12], but wanted [1].
      state = list(state) 
      state_hack = state
      state = tf.convert_to_tensor([state], dtype=tf.float32)
      actions = self.actor_main(state)
      ####   HACK
      dir_x = -3.396226415 - state_hack[0] 
      dir_y = -5.408805031 - state_hack[1] 
    
      actions = [dir_x,dir_y]
      
      actions =  tf.convert_to_tensor([actions], dtype=tf.float32)

       #####  HACK ENds
     

      
      # For training, we added noise in action and for testing, we will not add any noise.
      if not evaluate:  # if we are training then we want to get some random normal noise
          #### TODO:: We can over time reduce the stddev 
          actions += tf.random.normal(shape=[2], mean=0.0, stddev=0.1)
      # If o/p of our DNN is 1 or .999 and then we add noise (0.1) then we may get action
      # which may be outside the bound of my env which was boud by +/- 1 .
      # We will go ahead an clip this to make sure we dont pass any illigal action to the env.
      # Any values less than clip_value_min are set to clip_value_min. Any values greater than 
      # clip_value_max are set to clip_value_max. 
      actions = self.max_action * (tf.clip_by_value(actions, self.min_action, self.max_action))
      #print(actions)
      # Since this is tensor so we send the 0th element as the value
      #  is 0th element which is a numpy array
      return actions[0]  #<tf.Tensor: shape=(2,), dtype=float32, numpy=array([-0.10161366,  0.08597417], dtype=float32)>

  # Interface function for agent's memory
  def savexp(self,state, next_state, action, done, reward, accuracy):
        self.memory.storexp(state, next_state, action, done, reward,accuracy)

  # This is where we do hard copy of the initial weights of the actor/critic network
  #  to target actor/critic
  def update_target(self, tau=None):
      # We have to deal with the base case of the hard copy on the 
      #  first call on the function and soft copy on every other call of this funtion.
      if tau is None:  # if tau is not supplied then use the default defined in the constructor.
          tau = self.tau

      weights1 = []
      # target for actor
      #targets1 = self.actor_target.weights  # Huber weights are  array([0., 0.],
      targets1 = self.actor_target.get_weights()
      # We will iterate over the actor weights and append the weight for actor
      # multiplied by tau and add the weight if the target actor multiplied by (1-tau)
      #for i, weight in enumerate(self.actor_main.weights):
      for i, weight in enumerate(self.actor_main.get_weights()):
          weights1.append(weight * tau + targets1[i]*(1-tau))
      # After we go over the loop(every iteration) we set the weights of the target actor 
      # to the list of weights1. We are moving slowly target network towards the trained network 
      # To make sure that target network moves slow and same we do for critic network
      
      # HACK  since actor_main was not set therefore weights were not set
      #if weights1:
      self.actor_target.set_weights(weights1)

      # target for critic , we do the same for the critic network as explained above
      weights2 = []
      targets2 = self.critic_target.get_weights()
      for i, weight in enumerate(self.critic_main.get_weights()):
          weights2.append(weight * tau + targets2[i]*(1-tau))
      self.critic_target.set_weights(weights2)

      weights3 = []
      targets3 = self.critic_target2.get_weights()
      for i, weight in enumerate(self.critic_main2.get_weights()):
          weights3.append(weight * tau + targets3[i]*(1-tau))
      self.critic_target2.set_weights(weights3)


  # We have the learning function to learn where the bulk of the functionality comes in. 
  # We check if our memory is filled till Batch Size. We dont want to learn for less
  #  then batch size. Then only call train function.          
  def train(self, file_target_actions,file_target_next_state_values,file_target_next_state_values2,
               file_actor_loss,file_critic_value,file_critic_value2, file_critic_loss,file_critic_loss2,
               file_new_policy_actions,file_next_state_target_value, file_target_values,
               file_all_trained_states_R,agent):
      if self.memory.cnt < self.batch_size:
        return 
      target_actions = 0
      target_next_state_values = 0
      target_next_state_values2 = 0
      critic_value = 0
      critic_value2 = 0
      next_state_target_value = 0
      target_values = 0
      critic_loss1 = 0
      critic_loss2 = 0
      new_policy_actions = 0
      actor_loss  = 0

      # Sample our memory after batch size is filled.state, next_state, actionArray, done, reward,next_accuracy
      states, next_states, rewards, actions, dones, accuracy = self.memory.sample(self.batch_size)
      states_df = pd.DataFrame(states)

      embeddings_df = states_df[[0, 1]]

      embeddings_df.rename(columns={0: 'emb_x', 1: 'emb_y'}, inplace=True)
      #Hack adding critic value to the embeddings
      #embeddings_df = embeddings_df.reset_index()  # make sure indexes pair with number of rows
      early_critic_list_1 = []
      early_critic_list_2 = []
      early_critic_list_3 = []
      early_critic_list_4 = []
      early_critic_list_5 = []
      early_critic_list_6 = []
      early_critic_list_7 = []
      early_critic_list_8 = []
      early_critic_list_9 = []
      early_critic_list_10 = []
      early_critic_list_11 = []
      early_critic_list_12 = []
      early_critic_list_13 = []
      early_critic_list_14 = []
      early_critic_list_15 = []
      early_critic_list_16 = []
      early_critic_list_17 = []
      early_critic_list_18 = []
      early_critic_list_19 = []
      early_critic_list_20 = []
      early_critic_list_21 = []
      early_critic_list_22 = []
      early_critic_list_23 = []
      early_critic_list_24 = []
      
      mid_critic_list_1 = []
      mid_critic_list_2 = []
      mid_critic_list_3 = []
      mid_critic_list_4 = []
      mid_critic_list_5 = []
      mid_critic_list_6 = []
      mid_critic_list_7 = []
      mid_critic_list_8 = []
      mid_critic_list_9 = []
      mid_critic_list_10 = []
      mid_critic_list_11 = []
      mid_critic_list_12 = []
      mid_critic_list_13 = []
      mid_critic_list_14 = []
      mid_critic_list_15 = []
      mid_critic_list_16 = []
      mid_critic_list_17 = []
      mid_critic_list_18 = []
      mid_critic_list_19 = []
      mid_critic_list_20 = []
      mid_critic_list_21 = []
      mid_critic_list_22 = []
      mid_critic_list_23 = []
      mid_critic_list_24 = []
      
      critic_list = []
      for index, row in embeddings_df.iterrows():
        print(row[0], row[1])
        state = [0,0,0,1,0,0,0,0,0,0,0,0]
        state[0] = row[0] 
        state[1] = row[1]
        action = agent.act(state, True)
        c_states = np.array([state])
        c_actions = np.array([action])
        c_states = tf.convert_to_tensor(c_states, dtype= tf.float32)
        c_actions = tf.convert_to_tensor(c_actions, dtype= tf.float32)

        critic_early = agent.critic_main.early_call(c_states, c_actions)
        #critic_early = (tf.keras.backend.get_value(critic_early))[0][22]
        early_critic_list_1.append((tf.keras.backend.get_value(critic_early))[0][0])
        early_critic_list_2.append((tf.keras.backend.get_value(critic_early))[0][1])
        early_critic_list_3.append((tf.keras.backend.get_value(critic_early))[0][2])
        early_critic_list_4.append((tf.keras.backend.get_value(critic_early))[0][3])
        early_critic_list_5.append((tf.keras.backend.get_value(critic_early))[0][4])
        early_critic_list_6.append((tf.keras.backend.get_value(critic_early))[0][5])
        early_critic_list_7.append((tf.keras.backend.get_value(critic_early))[0][6])
        early_critic_list_8.append((tf.keras.backend.get_value(critic_early))[0][7])
        early_critic_list_9.append((tf.keras.backend.get_value(critic_early))[0][8])
        early_critic_list_10.append((tf.keras.backend.get_value(critic_early))[0][9])
        early_critic_list_11.append((tf.keras.backend.get_value(critic_early))[0][10])
        early_critic_list_12.append((tf.keras.backend.get_value(critic_early))[0][11])
        early_critic_list_13.append((tf.keras.backend.get_value(critic_early))[0][12])
        early_critic_list_14.append((tf.keras.backend.get_value(critic_early))[0][13])
        early_critic_list_15.append((tf.keras.backend.get_value(critic_early))[0][14])
        early_critic_list_16.append((tf.keras.backend.get_value(critic_early))[0][15])
        early_critic_list_17.append((tf.keras.backend.get_value(critic_early))[0][16])
        early_critic_list_18.append((tf.keras.backend.get_value(critic_early))[0][17])
        early_critic_list_19.append((tf.keras.backend.get_value(critic_early))[0][18])
        early_critic_list_20.append((tf.keras.backend.get_value(critic_early))[0][19])
        early_critic_list_21.append((tf.keras.backend.get_value(critic_early))[0][20])
        early_critic_list_22.append((tf.keras.backend.get_value(critic_early))[0][21])
        early_critic_list_23.append((tf.keras.backend.get_value(critic_early))[0][22])
        early_critic_list_24.append((tf.keras.backend.get_value(critic_early))[0][23])
        
        
        critic_mid = agent.critic_main.mid_call(c_states, c_actions)
        #critic_mid = (tf.keras.backend.get_value(critic_mid))[0]
        mid_critic_list_1.append((tf.keras.backend.get_value(critic_mid))[0][0])
        mid_critic_list_2.append((tf.keras.backend.get_value(critic_mid))[0][1])
        mid_critic_list_3.append((tf.keras.backend.get_value(critic_mid))[0][2])
        mid_critic_list_4.append((tf.keras.backend.get_value(critic_mid))[0][3])
        mid_critic_list_5.append((tf.keras.backend.get_value(critic_mid))[0][4])
        mid_critic_list_6.append((tf.keras.backend.get_value(critic_mid))[0][5])
        mid_critic_list_7.append((tf.keras.backend.get_value(critic_mid))[0][6])
        mid_critic_list_8.append((tf.keras.backend.get_value(critic_mid))[0][7])
        mid_critic_list_9.append((tf.keras.backend.get_value(critic_mid))[0][8])
        mid_critic_list_10.append((tf.keras.backend.get_value(critic_mid))[0][9])
        mid_critic_list_11.append((tf.keras.backend.get_value(critic_mid))[0][10])
        mid_critic_list_12.append((tf.keras.backend.get_value(critic_mid))[0][11])
        mid_critic_list_13.append((tf.keras.backend.get_value(critic_mid))[0][12])
        mid_critic_list_14.append((tf.keras.backend.get_value(critic_mid))[0][13])
        mid_critic_list_15.append((tf.keras.backend.get_value(critic_mid))[0][14])
        mid_critic_list_16.append((tf.keras.backend.get_value(critic_mid))[0][15])
        mid_critic_list_17.append((tf.keras.backend.get_value(critic_mid))[0][16])
        mid_critic_list_18.append((tf.keras.backend.get_value(critic_mid))[0][17])
        mid_critic_list_19.append((tf.keras.backend.get_value(critic_mid))[0][18])
        mid_critic_list_20.append((tf.keras.backend.get_value(critic_mid))[0][19])
        mid_critic_list_21.append((tf.keras.backend.get_value(critic_mid))[0][20])
        mid_critic_list_22.append((tf.keras.backend.get_value(critic_mid))[0][21])
        mid_critic_list_23.append((tf.keras.backend.get_value(critic_mid))[0][22])
        mid_critic_list_24.append((tf.keras.backend.get_value(critic_early))[0][23])
        
        critic = tf.squeeze(agent.critic_main(c_states, c_actions), 1)
        critic = (tf.keras.backend.get_value(critic))[0]
        
        critic_list.append(critic)
        prop = list(embeddings_df.columns.values)
      embeddings_df['critic'] = critic_list  

      embeddings_df['critic_early_1'] = early_critic_list_1
      embeddings_df['critic_early_2'] = early_critic_list_2
      embeddings_df['critic_early_3'] = early_critic_list_3
      embeddings_df['critic_early_4'] = early_critic_list_4
      embeddings_df['critic_early_5'] = early_critic_list_5
      embeddings_df['critic_early_6'] = early_critic_list_6
      embeddings_df['critic_early_7'] = early_critic_list_7
      embeddings_df['critic_early_8'] = early_critic_list_8
      embeddings_df['critic_early_9'] = early_critic_list_9
      embeddings_df['critic_early_10'] = early_critic_list_10
      embeddings_df['critic_early_11'] = early_critic_list_11
      embeddings_df['critic_early_12'] = early_critic_list_12
      embeddings_df['critic_early_13'] = early_critic_list_13
      embeddings_df['critic_early_14'] = early_critic_list_14
      embeddings_df['critic_early_15'] = early_critic_list_15
      embeddings_df['critic_early_16'] = early_critic_list_16
      embeddings_df['critic_early_17'] = early_critic_list_17
      embeddings_df['critic_early_18'] = early_critic_list_18
      embeddings_df['critic_early_19'] = early_critic_list_19
      embeddings_df['critic_early_20'] = early_critic_list_20
      embeddings_df['critic_early_21'] = early_critic_list_21
      embeddings_df['critic_early_22'] = early_critic_list_22
      embeddings_df['critic_early_23'] = early_critic_list_23
      embeddings_df['critic_early_24'] = early_critic_list_24
      
      embeddings_df['critic_mid_1'] = mid_critic_list_1
      embeddings_df['critic_mid_2'] = mid_critic_list_2
      embeddings_df['critic_mid_3'] = mid_critic_list_3
      embeddings_df['critic_mid_4'] = mid_critic_list_4
      embeddings_df['critic_mid_5'] = mid_critic_list_5
      embeddings_df['critic_mid_6'] = mid_critic_list_6
      embeddings_df['critic_mid_7'] = mid_critic_list_7
      embeddings_df['critic_mid_8'] = mid_critic_list_8
      embeddings_df['critic_mid_9'] = mid_critic_list_9
      embeddings_df['critic_mid_10'] = mid_critic_list_10
      embeddings_df['critic_mid_11'] = mid_critic_list_11
      embeddings_df['critic_mid_12'] = mid_critic_list_12
      embeddings_df['critic_mid_13'] = mid_critic_list_13
      embeddings_df['critic_mid_14'] = mid_critic_list_14
      embeddings_df['critic_mid_15'] = mid_critic_list_15
      embeddings_df['critic_mid_16'] = mid_critic_list_16
      embeddings_df['critic_mid_17'] = mid_critic_list_17
      embeddings_df['critic_mid_18'] = mid_critic_list_18
      embeddings_df['critic_mid_19'] = mid_critic_list_19
      embeddings_df['critic_mid_20'] = mid_critic_list_20
      embeddings_df['critic_mid_21'] = mid_critic_list_21
      embeddings_df['critic_mid_22'] = mid_critic_list_22
      embeddings_df['critic_mid_23'] = mid_critic_list_23
      embeddings_df['critic_mid_24'] = mid_critic_list_24
      
      embeddings_df.to_csv(file_all_trained_states_R,index=False)   
      file_all_trained_states_R.flush()
      
   



      # convert states and new states, rewards and actions to tensor.
      states = tf.convert_to_tensor(states, dtype= tf.float32)
      next_states = tf.convert_to_tensor(next_states, dtype= tf.float32)
      rewards = tf.convert_to_tensor(rewards, dtype= tf.float32)
      actions = tf.convert_to_tensor(actions, dtype= tf.float32)
      accuracy = tf.convert_to_tensor(accuracy, dtype= tf.float32)
      # we dont have to do terminal flags (dones) to tensor as will be only
      #  doing numpy operations on them.
      # dones = tf.convert_to_tensor(dones, dtype= tf.bool)

      # use gradient of tape for calculations for our gradients. The gradient tape is used 
      # to load up operations to computational graph for calculations of gradients.
      # This way when we call choose action act() function on Agent network. Those operations are 
      # are not stored anywhere that is used for calculate of gradients. So it is effectively detached 
      # from the graph. So only things within this context manager are used for the 
      # calculation of our gradient. This is where we stick the update rule.

      # Lets go ahead and start with the critic network. We update critic 
      # by minimizing the loss. We have to take the new state and pass it by the 
      # target actor network and then get the target critic's evaluation of the new 
      # state's and those target action's and then we can calculate the target
      #  which is reward + gamma multiplied by the critic value of the new state times 
      #  1 minus terminal flag. Then take mean sq error between the target value and the 
      # critic value for the states and actions the agent actually took. 
      

      with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
          # The target value for critic loss is calculated by predicting action 
          # for the next states using the actor’s target network and then using 
          # these actions we get the next state’s values using the critic’s target network.
          # target_actions is the target actor what are the things 
          # we should do for the new states
          # To avoid the looping issue it uses actor_target and keep actor_main stable
          target_actions = self.actor_target(next_states)
          print('############################### target_actions from actor_target ########################################')
          print(tf.keras.backend.get_value(target_actions))


          target_actions += tf.clip_by_value(tf.random.normal(shape=[*np.shape(target_actions)], mean=0.0, stddev=0.2), -0.5, 0.5)
          target_actions = self.max_action * (tf.clip_by_value(target_actions, self.min_action, self.max_action))
          print('############################### target_actions after clip and updates ########################################')
          file_target_actions.write(np.array2string(tf.keras.backend.get_value(target_actions), precision=8, separator=','))
          
          # critic value for new states is shown below as : target critic evaluation 
          # of the next states and target actions and squeeze along the first dimension.
          # It does the forward pass. 
          # The value of the succesor state for the best action

          target_next_state_values = tf.squeeze(self.critic_target(next_states, target_actions), 1)
          print('############################### target_next_state_values ########################################')
          file_target_next_state_values.write(np.array2string(tf.keras.backend.get_value(target_next_state_values), precision=8, separator=','))  
          target_next_state_values2 = tf.squeeze(self.critic_target2(next_states, target_actions), 1)
          print('############################### target_next_state_values2 ########################################')
         
          file_target_next_state_values2.write(np.array2string(tf.keras.backend.get_value(target_next_state_values2), precision=8, separator=','))

          # Our predicted (critic) values are the output of the main critic network which takes
          #  states and actions from the buffer sample.
          # critic value is the value of the current state's with respect to original state
          # and actions the agent actually took during the course of this episode.
          # This gives value of the action in a given state.
 
          critic_value = tf.squeeze(self.critic_main(states, actions), 1)
          print('############################### critic_value ########################################')
          print(tf.keras.backend.get_value(critic_value)) 
          file_critic_value.write(np.array2string(tf.keras.backend.get_value(critic_value), precision=8, separator=','))  
          critic_value2 = tf.squeeze(self.critic_main2(states, actions), 1)

          print('############################ critic_value2 ########################################')
          print(tf.keras.backend.get_value(critic_value2))  
          file_critic_value2.write(np.array2string(tf.keras.backend.get_value(critic_value2), precision=8, separator=','))  
          next_state_target_value = tf.math.minimum(target_next_state_values, target_next_state_values2)
          print('############################ next_state_target_value ########################################')
          print(tf.keras.backend.get_value(next_state_target_value))  

          file_next_state_target_value.write(np.array2string(tf.keras.backend.get_value(next_state_target_value), precision=8, separator=','))

          # target for the terminal new state is just the reward for every other state
          # it is reward + discounted value of the resulting state according to the target critic network.
          # Then, we apply the Bellman equation to calculate target values 
          # (target_values = rewards + self.gamma * target_next_state_values * done)
          # If done (0) then there is no succesor state then the target value is the reward.
          target_values = rewards + self.gamma * next_state_target_value * dones
          

          
          print('############################ target_values = rewards + self.gamma * next_state_target_value * dones ########################################')
          print(tf.keras.backend.get_value(target_values))    
          file_target_values.write(np.array2string(tf.keras.backend.get_value(target_values), precision=8, separator=','))         

          # The loss function computed out of three network,actor_target, critic_target & critic_main
          # Critic loss is then calculated as MSE of target values and predicted values.
          # Boltzmann error is the diff (MSE) between the value of the action in the current state and
          # (Reward + gamma times max of all possible action of the value in the successor state )

          critic_loss1 = tf.keras.losses.MSE(target_values, critic_value)
          critic_loss2 = tf.keras.losses.MSE(target_values, critic_value2)

          print('############################ critic_loss1 ########################################')
          print(tf.keras.backend.get_value(critic_loss1)) 
          file_critic_loss.write(np.array2string(tf.keras.backend.get_value(critic_loss1), precision=8, separator=',')+ "\n")  

          print('############################ critic_loss2 ########################################')
          print(tf.keras.backend.get_value(critic_loss2)) 
          file_critic_loss2.write(np.array2string(tf.keras.backend.get_value(critic_loss2), precision=8, separator=',')+ "\n")  

      # for critic loss calculate gradients and apply the gradients 
      # Compute grad with respect to critic_main 
      grads1 = tape1.gradient(critic_loss1, self.critic_main.trainable_variables)
      grads2 = tape2.gradient(critic_loss2, self.critic_main2.trainable_variables)

      # Apply our gradients on same critic_main trainable_variables 
      # All the weights of the three alyers in the  main critic network
      # Gradient involves all the weights of all three networks and we 
      # apply only to the main critic network weights of all layers only
      self.c_opt1.apply_gradients(zip(grads1, self.critic_main.trainable_variables))
      self.c_opt2.apply_gradients(zip(grads2, self.critic_main2.trainable_variables))

      self.trainstep +=1
      # To make sure actor gets updated less times critic
      if self.trainstep % self.actor_update_steps == 0:
                
          with tf.GradientTape() as tape3:
            # These are the actions according to actor based upon its current 
            # set of weights. Not based upon the weights it had at the time 
            # whatever the memory we stored in a agent's memory
            new_policy_actions = self.actor_main(states)

            print('############################ new_policy_actions ########################################')
            print(tf.keras.backend.get_value(new_policy_actions))
            file_new_policy_actions.write(np.array2string(tf.keras.backend.get_value(new_policy_actions), precision=8, separator=','))

            # It is negative ('-') as we are doing gradient ascent.
            # As in policy gradient methods we dont want to use decent 
            # because that will minimize the total score over time. 
            # Rather we would like to maximize the total score over time.
            # Gradient ascent is just negative of gradient decent.
            # Actor loss is calculated as negative of critic main values with 
            # inputs as the main actor predicted actions.

            #                       actor_loss = -self.critic_main(states, new_policy_actions) 
            # Then our loss is reduce mean of that actor loss.
            #                   actor_loss = tf.math.reduce_mean(actor_loss)

            #######  HACK
            ####   
            state_hack = [0,0,0,2,0,0,0,0,0,0,0,0]
            state_hack[0] = -3.396226415  
            state_hack[1] = -5.408805031
            state_hack = tf.convert_to_tensor([state_hack], dtype=tf.float32)
            states_diff = tf.subtract(state_hack,states)
            # Strip first 64 (embx,emby)
            actions = states_diff[:,0:2]
            print(tf.keras.backend.get_value(actions))
            actions = 1 * (tf.clip_by_value(actions, -1, 1))
            print(tf.keras.backend.get_value(actions))
            actor_loss = K.mean(K.square( actions - new_policy_actions)) 

            ######## HACK END

            print('############################ actor_loss ########################################')
            print(tf.keras.backend.get_value(actor_loss))   
            file_actor_loss.write(np.array2string(tf.keras.backend.get_value(actor_loss), precision=8, separator=',') + "\n")            

            # In the paper they applied the chain rule gradient of critic network with actor network
            # This is how we get the gradient of the critic loss with respect to Meu (µ) parameter 
            # by taking this actor loss which is proportional to the output of the critic network
            # and is coupled. The gradient is non zero because it has this dependency on the output of our 
            # actor networks. Dependence bacause of non-zero gradient comes from the fact that we are taking actions
            #  with respect to actor network, which is calculated according to theatas (Ɵ super µ)
            #  That can effect from here to the critic network.  That's what allows to take the gradient of the output of the 
            # critic network with respect to the variables of the actor network that's how we get coupling.

          # Since actor_loss involves actor_main & critic_main
          grads3 = tape3.gradient(actor_loss, self.actor_main.trainable_variables)

          ## HACK print the weights for actor_main
          # Apply the gradients to actor_main trainable variables.
          self.a_opt.apply_gradients(zip(grads3, self.actor_main.trainable_variables))

      #if self.trainstep % self.replace == 0:
      # Perform soft update on our target network. We use default value of tau = 0.005
      # we update our target networks with a tau of 0.005.
      self.update_target()  
      return target_actions,target_next_state_values,target_next_state_values2,critic_value,critic_value2,next_state_target_value,target_values,critic_loss1,critic_loss2,new_policy_actions,actor_loss   

