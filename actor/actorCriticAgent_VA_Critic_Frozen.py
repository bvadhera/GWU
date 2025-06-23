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
        self.state_h3 = Dense(24, activation=LeakyReLU(alpha=0.1), name='state_h3')
        self.critic_output_A = Dense(1, activation='tanh', name='criticA_output')
        self.critic_output_V = Dense(1, activation='tanh', name='criticV_output')
    #  We have call function for Critic for forward propagation operation.
    # x outputs the V function with state and y outputs V function with state and action
    def call(self, predict_layer, action):
        #x = self.state_h1(tf.concat([predict_layer, action], axis=1))
        x = self.state_h1(tf.concat([predict_layer], axis=1))
        y = self.state_h2(tf.concat([x, action], axis=1))
        y = self.critic_output_A(y)
        x = self.state_h3(x)
        x = self.critic_output_V(x)
        return x,y

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
      
      critic_list_A = []
      critic_list_V = []
      for index, row in embeddings_df.iterrows():
        print(row[0], row[1])
        state = [0,0,0,1,0,0,0,0,0,0,0,0]
        state[0] = row[0] 
        state[1] = row[1]
        action = agent.actor_main(state, False)
        c_states = np.array([state])
        c_actions = np.array([action])
        c_states = tf.convert_to_tensor(c_states, dtype= tf.float32)
        c_actions = tf.convert_to_tensor(c_actions, dtype= tf.float32)

         
        critic_A_V = tf.squeeze(agent.critic_main(c_states, c_actions), 2)
        critic_A = (tf.keras.backend.get_value(critic_A_V))[0][0]
        critic_V = (tf.keras.backend.get_value(critic_A_V))[1][0]
        critic_list_A.append(critic_A)
        critic_list_V.append(critic_V)
        #prop = list(embeddings_df.columns.values)
      embeddings_df['critic_A'] = critic_list_A  
      embeddings_df['critic_V'] = critic_list_V  
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

           # This is an
            actor_loss = -self.critic_main(states, new_policy_actions)[1] 
            # Then our loss is reduce mean of that actor loss.
            actor_loss = tf.math.reduce_mean(actor_loss)
 

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

