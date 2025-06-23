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
    self.action_memory = np.zeros((maxsize, naction), dtype=np.float32)   #RAGHAV - changed naction = 3
    # we will need an reward memory which is just shape memory size. It will be 
    # just an array of floating point numbers.
    self.reward_memory = np.zeros((maxsize,), dtype=np.float32)
    # just an array of floating point numbers.
    self.accuracy_memory = np.zeros((maxsize,), dtype=np.float32)
    # just an array of int numbers.
    self.depth_memory = np.zeros((maxsize,), dtype=np.int32)
    #  We also need a terminal memory which will be of type boolean 
    self.done_memory = np.zeros((maxsize,), dtype= np.bool)

  # We need a function to store transitions where transition is 
  #   state,action,reward, new_state(Next_state) and terminal flag (done)
  def storexp(self, state, next_state, action, done, reward, accuracy,depth):
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
    self.depth_memory[index] = depth
    # Increment mem counter by 1
    self.cnt += 1

  '''
  # We need a function to  sample our buffer
  def sample(self, count, batch_size):
    # We want to know how much of the memory we have filled up
    max_mem = min(count, self.maxsize)
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
    '''
      # We need a function to  sample our buffer
  def sample(self,batch):
    # Now we go ahead and de-reference our numpy arrays and return those at the end.
    states = self.state_memory[batch]
    next_states = self.next_state_memory[batch]
    rewards = self.reward_memory[batch]
    actions = self.action_memory[batch]
    dones = self.done_memory[batch]
    accuracy = self.accuracy_memory[batch]
    depth = self.depth_memory[batch]
    return states, next_states, rewards, actions, dones,accuracy, depth


#Raghav
def lambda_function(inputV):
  return tf.keras.utils.normalize(inputV, axis=-1, order=2)

# Actor  network class
class Actor(tf.keras.Model):
    def __init__(self):
        super(Actor, self).__init__()    
        self.h1 = Dense(10, activation=LeakyReLU(alpha=0.1),  name='h1')
        self.h2 = Dense(10, activation=LeakyReLU(alpha=0.1),  name='h2')

        self.actor_output_X = Dense(2, activation='tanh', name='actor_output_X')

        self.actor_output_L = Lambda(lambda_function, name="lambda_layer")
        
        self.actor_output_D = Dense(1, activation='sigmoid', name='actor_output_D')
        
    #  We have call function for Actor for forward propagation operation.
    def call(self, predict_layer):
        x  = self.h1(predict_layer)
        x  = self.h2(x)
        xy = self.actor_output_X(x)
        ll = self.actor_output_L(xy)
        d  = self.actor_output_D(x) 
        x =  tf.concat([ll,d], axis=1)
        return x

    def mid_call(self, predict_layer):
        x  = self.h1(predict_layer)
        x  = self.h2(x)
        xy = self.actor_output_X(x)
        return xy
    
    def pre_call(self, predict_layer):
        x  = self.h1(predict_layer)
        y  = self.h2(x)
        z =  tf.concat([x,y], axis=1)
        return z
            
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
        y = self.critic_output_A(y) # Advantage Critic[1]
        x = self.state_h3(x)
        x = self.critic_output_V(x)   #Value Critic[0]
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
    self.n_actions = 3  # As actor gives three dimension x,y,z output.  # RAGHAV
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
    self.memory = RBuffer(3010001,  observation_space_shape, 3)
    # TO Save forward tree search memory  original was 3010001
    self.step_memory = RBuffer(3010001,  observation_space_shape, 3)
    self.trainstep = 0
    self.replace = 5
    # We need gamma - a discount factor for update equation
    #self.gamma =0.999   #RAGHAV
    self.gamma =0.0   #RAGHAV

    # max/min actions for our environment 
    # Huber What are min/max (-1/+1) ?
    # Ask Huber - what are they in our case
    self.min_action = -1.0  #RAGHAV
    self.max_action = 1.0   #RAGHAV
    self.min_action_dist = 0.0  #RAGHAV
    self.actor_update_steps = 2
    self.warmup = 2250000
    # default value for our soft update tau
    self.tau = 0.005   #RAGHAV
    # Note that we have compiled our target networks as we don’t want
    #  to get an error while copying weights from main networks to target networks.
    self.actor_target.compile(optimizer=self.a_opt)
    self.critic_target.compile(optimizer=self.c_opt1)
    self.critic_target2.compile(optimizer=self.c_opt2)

  def getGamma(self):
    return self.gamma

  # Choose an Action. It will take current state of environment as
  #  input as well as evaluate=False to train vs test. 
  #  Just test agent without adding the noise to get pure deterministic output.
  def act(self, state, evaluate=True):
    
      if self.trainstep > self.warmup:
            evaluate = True
    # For action selection, first, we convert our state into a tensor and then pass it
    #  to the actor-network.
    # Therefore we convert the state to tensor & add extra dimension to our 
    # observation(state) the batch dimension that is what deep NN expect as input.
    # They expect the batch dimension.
    #  np.asarray 
      #print ("state type", type(state))

      state = list(state) 
      state_hack = state
      state = tf.convert_to_tensor([state], dtype=tf.float32)
      #In case of not fixed polciy
      #actions = self.actor_main(state)
     
      ####   HACK  #RAGHAV
      dir_X = -3.396226415 - state_hack[0] 
      dir_Y = -5.408805031 - state_hack[1]
      dir_x = tf.convert_to_tensor([[dir_X/math.sqrt(dir_X**2+dir_Y**2)]], dtype=tf.float32)

      dir_y = tf.convert_to_tensor([[dir_Y/math.sqrt(dir_X**2+dir_Y**2)]], dtype=tf.float32)
      dist = tf.convert_to_tensor([[math.sqrt(dir_X**2+dir_Y**2)]], dtype=tf.float32)
      dist = self.max_action * (tf.clip_by_value(dist, self.min_action_dist, self.max_action))
      actions = tf.concat([dir_x, dir_y, dist], axis=1)
      actions_array = tf.keras.backend.get_value(actions)
      sin = tf.convert_to_tensor(np.array([[actions_array[0][1]]]), dtype=tf.float32)
      cos = tf.convert_to_tensor(np.array([[actions_array[0][0]]]), dtype=tf.float32)
      
      if not evaluate: #Explore
        # For training, we added noise in action and for testing, we will not add any noise.
        # if we are training then we want to get some random normal noise
        # Apply noise
        arct = tf.math.atan2(sin,cos)
        noise = tf.random.normal(shape=[1,1], mean=0.0, stddev=0.1)
        arct = arct + noise
        dist = self.max_action * (tf.clip_by_value(dist+noise, self.min_action_dist, self.max_action))
        cos = tf.math.cos(arct)
        sin = tf.math.sin(arct)
      actions = tf.concat([cos, sin, dist], axis=1)
      #####  HACK ENds
       

      #actions += tf.random.normal(shape=[2], mean=0.0, stddev=0.1)
      # If o/p of our DNN is 1 or .999 and then we add noise (0.1) then we may get action
      # which may be outside the bound of my env which was boud by +/- 1 .
      # We will go ahead an clip this to make sure we dont pass any illigal action to the env.
      # Any values less than clip_value_min are set to clip_value_min. Any values greater than 
      # clip_value_max are set to clip_value_max. 
      # CLIP DISTANCE PART OF THE actions only 
  
      #print(actions)
      # Since this is tensor so we send the 0th element as the value
      #  is 0th element which is a numpy array
      return actions[0] #<tf.Tensor: shape=(2,), dtype=float32, numpy=array([-0.10161366,  0.08597417], dtype=float32)>

  # Interface function for agent's memory 
  def savexp(self,state, next_state, action, done, reward, accuracy, depth):
        self.memory.storexp(state, next_state, action, done, reward,accuracy,depth)

  # Interface function for agent's memory from tree forward search 
  def savstepexp(self,state, next_state, action, done, reward, accuracy, depth):
        self.step_memory.storexp(state, next_state, action, done, reward,accuracy,depth)


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
          if targets1: #HUBER gammo-BasicTest
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

      #sampleSize = 150
      sampleSize = 300
      # Sample our memory after batch size is filled such as state, next_state, actionArray, done, reward,next_accuracy
      
      # For one step direct sample choose randomly from memory 150
      # For one step case we need a sample of 150 points from anything but last 50 episods i.e. 50x300 = 15000 points
      episode50Count = 15000  # 50 episods
      if self.memory.cnt <= episode50Count:
        # We want to know how much of the memory we have filled up
        max_mem = min(self.memory.cnt, self.memory.maxsize)
        # max_mem = min(self.cnt, self.maxsize)
        # Now we take batch of numbers,  replace= False as
        #  once the memory is sampled from that range it will not be sampled again.
        # it prevents you from double sampling again same memory
        batch = np.random.choice(max_mem, sampleSize, replace= False)  
        print("===============================batch============================================")
        print(batch)
        states, next_states, rewards, actions, dones, accuracy, depth = self.memory.sample(batch)
      else:
        truncCount = self.memory.cnt - episode50Count
        if truncCount > episode50Count :
                sampleCount = truncCount
        else :
                sampleCount = self.memory.cnt - truncCount
        # We want to know how much of the memory we have filled up
        max_mem = min(sampleCount, self.memory.maxsize)
        # max_mem = min(self.cnt, self.maxsize)
        # Now we take batch of numbers,  replace= False as
        #  once the memory is sampled from that range it will not be sampled again.
        # it prevents you from double sampling again same memory
        batch = np.random.choice(max_mem, sampleSize, replace= False)  
        states, next_states, rewards, actions, dones, accuracy, depth = self.memory.sample(batch)
      ''' 
      # For multi step Tree direct sample choose randomly from memory 150
      # For multi step Tree case we need a sample of 150 points from  last 50 episods i.e. 50x300 = 15000 points
      if self.memory.cnt <= episode50Count:
        max_mem = min(self.memory.cnt, self.memory.maxsize)
        batch = np.random.choice(max_mem, sampleSize, replace= False)  
        step_states, step_next_states, step_rewards, step_actions, step_dones, step_accuracy, step_depth = self.step_memory.sample(batch)
      else:
        trunc = self.memory.cnt - episode50Count
        max_mem=np.arange(trunc,self.memory.cnt)
        batch = np.random.choice(max_mem, sampleSize, replace= False)
        step_states, step_next_states, step_rewards, step_actions, step_dones, step_accuracy, step_depth = self.step_memory.sample(batch)
       
      
      # Now Concatinate both samples CHECK CODE YOU MAY NEED TO DO TENSOR CONCAT also

 
      states = np.concatenate((states, step_states), axis=0)
      next_states = np.concatenate((next_states, step_next_states), axis=0)
      rewards = np.concatenate((rewards, step_rewards), axis=0)
      actions = np.concatenate((actions, step_actions), axis=0)
      dones = np.concatenate((dones, step_dones), axis=0)      
      accuracy = np.concatenate((accuracy, step_accuracy), axis=0)   
      depth = np.concatenate((depth, step_depth), axis=0)   
      '''

      
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
        action = agent.act(state, True)  #HUBER gammo-BasicTest  Ask Dr Huber ?
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

      next_states_old = list(next_states)

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



          #target_actions = self.actor_target(next_states)
          target_actions = next_states  #HUBER gammo-BasicTest
          #Hack #HUBER gammo-BasicTest
          for i, val in enumerate(next_states_old):
            target_action = self.act(val)
            target_action = tf.keras.backend.get_value(target_action)
            target_action = list(target_action)
            target_action = tf.convert_to_tensor([target_action], dtype=tf.float32)
            if i != 0:
              target_actions = tf.concat([target_actions,target_action], axis=0)
            else:
              target_actions = target_action

          #HUBER gammo-BasicTest



          if True : #Print Statements
            print('############################### target_actions from actor_target ########################################')
            print(tf.keras.backend.get_value(target_actions))
          ''' #HUBER gammo-BasicTest
          # Get all three elements of target_actions                     
          dir_X = target_actions[:,0:1]
          dir_Y = target_actions[:,1:2]
          dist = target_actions[:,2:3]
          sin = dir_Y
          cos = dir_X
          # # Get arctan and add noise  
          arct = tf.math.atan2(sin,cos)
          noise = tf.random.normal(shape=[1,1], mean=0.0, stddev=0.1)
          arct = arct + noise
          # Clip distance between 0.0 and 1
          dist = self.max_action * (tf.clip_by_value(dist+noise, self.min_action_dist, self.max_action))
          cos = tf.math.cos(arct)
          sin = tf.math.sin(arct)

          target_actions = tf.concat([cos, sin, dist], axis=1)
          ''' #HUBER gammo-BasicTest
          #target_actions += tf.clip_by_value(tf.random.normal(shape=[*np.shape(target_actions)], mean=0.0, stddev=0.2), -0.5, 0.5)
          #target_actions = self.max_action * (tf.clip_by_value(target_actions, self.min_action, self.max_action))
          if True : #Print Statements
            print('############################### target_actions after clip and updates ########################################')
            file_target_actions.write(np.array2string(tf.keras.backend.get_value(target_actions), precision=8, separator=','))

          # critic value for new states is shown below as : target critic evaluation 
          # of the next states and target actions and squeeze along the first dimension.
          # It does the forward pass. 
          # The value of the succesor state for the best action

          target_state_values = tf.squeeze(self.critic_target(states, actions), 2)
          target_state_values2 = tf.squeeze(self.critic_target2(states, actions), 2)
          current_state_target_value = tf.math.minimum(target_state_values, target_state_values2)


          target_next_state_values = tf.squeeze(self.critic_target(next_states, target_actions), 2)
          if True : #Print Statements
            print('############################### target_next_state_values ########################################')
            file_target_next_state_values.write(np.array2string(tf.keras.backend.get_value(target_next_state_values), precision=8, separator=','))  
          target_next_state_values2 = tf.squeeze(self.critic_target2(next_states, target_actions), 2)
          if True : #Print Statements
            print('############################### target_next_state_values2 ########################################')
            file_target_next_state_values2.write(np.array2string(tf.keras.backend.get_value(target_next_state_values2), precision=8, separator=','))
          next_state_target_value = tf.math.minimum(target_next_state_values, target_next_state_values2)
          # Our predicted (critic) values are the output of the main critic network which takes
          #  states and actions from the buffer sample.
          # critic value is the value of the current state's with respect to original state
          # and actions the agent actually took during the course of this episode.
          # This gives value of the action in a given state.
 
          critic_value = tf.squeeze(self.critic_main(states, actions), 2)
          if True : #Print Statements
            print('############################### critic_value ########################################')
            print(tf.keras.backend.get_value(critic_value)) 
            critic_value_state = (tf.keras.backend.get_value(critic_value))[0][0]
            critic_value_state_action = (tf.keras.backend.get_value(critic_value))[1][0]
            file_critic_value.write(str(tf.keras.backend.get_value(critic_value_state))+','+ str(tf.keras.backend.get_value(critic_value_state_action))   + "\n")         
              
          
          critic_value2 = tf.squeeze(self.critic_main2(states, actions), 2)
          if True : #Print Statements
              print('############################ critic_value2 ########################################')
              print(tf.keras.backend.get_value(critic_value2))  
              critic_value2_state = (tf.keras.backend.get_value(critic_value2))[0][0]
              critic_value2_state_action = (tf.keras.backend.get_value(critic_value2))[1][0]
              file_critic_value.write(str(tf.keras.backend.get_value(critic_value2_state))+','+ str(tf.keras.backend.get_value(critic_value2_state_action))   + "\n")         
              file_critic_value2.write(np.array2string(tf.keras.backend.get_value(critic_value2), precision=8, separator=','))  
              print('############################ next_state_target_value ########################################')
              print(tf.keras.backend.get_value(next_state_target_value))  
              file_next_state_target_value.write(np.array2string(tf.keras.backend.get_value(next_state_target_value), precision=8, separator=','))

          # target for the terminal new state is just the reward for every other state
          # it is reward + discounted value of the resulting state according to the target critic network.
          # Then, we apply the Bellman equation to calculate target values 
          # (target_values = rewards + self.gamma * target_next_state_values * done)
          # If done (0) then there is no succesor state then the target value is the reward.
          #target_values = rewards + self.gamma * next_state_target_value * dones
          
          # To get stable values we will use Advantage Actor - critic learning
          #real depth of the tree   depth + 1
          #Value Critic[0]  #Advantage Critic[1]
          depth = depth + 1 #including the root node
          gammaU = self.gamma ** depth
   
          target_values1  = rewards + gammaU * ((next_state_target_value)[0]) - ((current_state_target_value)[0])  # Q-V (Target for Advantage Value)
          target_values2  =  rewards + gammaU * ((next_state_target_value)[0]) #Target Value  for V
          print('############################ target_values = rewards + self.gamma * next_state_target_value * dones ########################################')
          #print(tf.keras.backend.get_value(target_values))    

          #file_target_values.write("target_values1") 
          #file_target_values.write(np.array2string(tf.keras.backend.get_value(target_values1), precision=8, separator=','))         
          #file_target_values.write("target_values2") 
          #file_target_values.write(np.array2string(tf.keras.backend.get_value(target_values2), precision=8, separator=','))         

          # The loss function computed out of three network,actor_target, critic_target & critic_main
          # Critic loss is then calculated as MSE of target values and predicted values.
          # Boltzmann error is the diff (MSE) between the value of the action in the current state and
          # (Reward + gamma times max of all possible action of the value in the successor state )

          #critic_loss1 = tf.keras.losses.MSE(target_values, critic_value)
          #critic_loss2 = tf.keras.losses.MSE(target_values, critic_value2)

          boosting1 =  50  #RAGHAV
          boosting2 = 1
          critic_loss11 = tf.keras.losses.MSE((critic_value[1]) ,(boosting1*(target_values1)) )   # Loss for Advantage Value
          critic_loss12 = boosting2 * (tf.keras.losses.MSE((critic_value[0]) , (target_values2))) # Loss for State Value
          critic_loss1 = critic_loss11 + critic_loss12   #HUBER
          #critic_loss1 =  critic_loss12

          if True : #Print Statements
              print('############################ critic_loss1 ########################################')
              print(tf.keras.backend.get_value(critic_loss11)) 
              print(tf.keras.backend.get_value(critic_loss12)) 
              print(tf.keras.backend.get_value(critic_loss1))
              file_critic_loss.write(str(tf.keras.backend.get_value(critic_loss11))+','+ str(tf.keras.backend.get_value(critic_loss12)) +','+ str(tf.keras.backend.get_value(critic_loss1))  + "\n")         
          
          critic_loss21 = tf.keras.losses.MSE((critic_value2[1]) ,(boosting1*(target_values1)) )  # Loss for Advantage Value
          #critic_loss22 = boosting2 * (tf.keras.losses.MSE((current_state_target_value[0]) , (target_values2))) # Loss for State Value
          critic_loss22 = boosting2 * (tf.keras.losses.MSE((critic_value2[0]) , (target_values2))) # Loss for State Value
          critic_loss2 = critic_loss21 + critic_loss22   #HUBER
          #critic_loss2 = critic_loss22
          if True : #Print Statements
              print('############################ critic_loss2 ########################################')
              print(tf.keras.backend.get_value(critic_loss21)) 
              print(tf.keras.backend.get_value(critic_loss22)) 
              print(tf.keras.backend.get_value(critic_loss2))
              file_critic_loss2.write(str(tf.keras.backend.get_value(critic_loss21))+','+ str(tf.keras.backend.get_value(critic_loss22)) +','+ str(tf.keras.backend.get_value(critic_loss2))  + "\n")         
                
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
            new_policy_actions = self.actor_main(states) #HUBER gammo-BasicTest  Ask Dr Huber ?
            if True : #Print Statements
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
            ## How do we calculate distance from different # ==============================Ask Dr Huber-------
            dir_X = states_diff[:,0:1]
            dir_Y = states_diff[:,1:2]
            #RAGHAV
            dir_x = dir_X/tf.sqrt(dir_X**2+dir_Y**2)
            dir_y =  dir_Y/tf.sqrt(dir_X**2+dir_Y**2) 
            # No need to add noise 
            # No need to clip dir_x  and dir_y as they should be between -1 and 1
            dist = tf.sqrt(dir_X**2+dir_Y**2)
            # clip distance 0, 1
            dist = self.max_action * (tf.clip_by_value(dist, self.min_action_dist, self.max_action))
            actions = tf.concat([ dir_x,dir_y,dist], axis=1)
 
            #actions = 1 * (tf.clip_by_value(actions, -1, 1)) # ===========#RAGHAV=================Ask Dr Huber-------
 
            actor_loss = K.mean(K.square( actions - new_policy_actions)) 

            ######## HACK END
            if True : #Print Statements
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

