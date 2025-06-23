import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
import pandas as pd
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
#from tensorflow.python.keras.backend import set_session
#from tensorflow.python.keras.models import load_model

tf.compat.v1.disable_eager_execution()
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)

tf.random.set_seed(934733)

avg_rewards_list = []

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu,True)
    


def normal_round(n):
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)

def custom_loss(y_true, y_pred):
    print("type(y_true)")
    print(type(y_true) )
    print("type(y_pred)")
    print(type(y_pred) )
    # convert y_true to tensorflow.python.keras.engine.keras_tensor.KerasTensor
    print ('##################################')  # shape 
    print ('y_true.shape:', y_true.shape)  # shape 
    print ('y_pred.shape:',y_pred.shape)   # shape 
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    #def loss(y_true,y_pred):
    weight = 0.1  
    loss =  K.mean(K.square( y_true[:,1:4] - y_pred[:,1:4]), axis=-1)  + weight*(K.square(y_true[:,:1] - y_pred[:,:1]))
    return loss

def custom_loss_actor(y_true, y_pred):
    print("type(y_true)")
    print(type(y_true) )
    print("type(y_pred)")
    print(type(y_pred) )
    # convert y_true to tensorflow.python.keras.engine.keras_tensor.KerasTensor
    print ('##################################')  # shape 
    print ('y_true.shape:', y_true.shape)  # shape 
    print ('y_pred.shape:',y_pred.shape)   # shape 
    # Return a function
    return 0*(K.mean(K.square( y_true - y_pred), axis=-1) )

def custom_loss_critic(y_true, y_pred):
    print("type(y_true)")
    print(type(y_true) )
    print("type(y_pred)")
    print(type(y_pred) )
    # convert y_true to tensorflow.python.keras.engine.keras_tensor.KerasTensor
    print ('##################################')  # shape 
    print ('y_true.shape:', y_true.shape)  # shape 
    print ('y_pred.shape:',y_pred.shape)   # shape 
    # Return a function
    return 0*(K.mean(K.square( y_true - y_pred), axis=-1) )

# Accuracy Matrics First
def decoder_accuracy(y_true,y_pred):
    #predx = K.batch_get_value(y_pred[:,:3])  # tensor for num_layers, node1 & node2 after decoder
    #truex = K.batch_get_value(y_true[:,:3]) # tensor for num_layers, node1 & node2 before decoder
    denorm_y_true = y_true[:,1:4] * devideBy 
    denorm_y_pred = y_pred[:,1:4] * devideBy 

    diff = K.equal(K.mean(K.round(denorm_y_true - denorm_y_pred),  axis=-1), 0)
    return(K.cast(diff,tf.float32))

# Accuracy Matrics Second
def mean_sqe_pred(y_true,y_pred): 

    denorm_y_true = y_true[:,:1]
    denorm_y_pred = y_pred[:,:1] 
    return K.square(denorm_y_true - denorm_y_pred)

# Accuracy Matrics First
def actor_accuracy(y_true,y_pred):
    return(K.zeros(1))

# Accuracy Matrics First
def critic_accuracy(y_true,y_pred):
    return(K.zeros(1))

def getMaxNumOfNodesInDataSet(df_max):
    mLayers = df_max['num_layers'].max()
    # iterate as many layers and create node columns 
    i = 0
    maxNumOfNodes = []
    for i in range(mLayers):
      node = "num_node" + str(i+1)
      attrib = df_max[node]
      maxNumOfNodes.append(int(attrib.max()))
    print (max(maxNumOfNodes))
    #get max number of nodes
    return max(maxNumOfNodes)

'''
def normalize(df_,features):
  #result = df.copy()
  for feature_name in features:
    max_value = df_[feature_name].max()
    min_value = df_[feature_name].min() ;print("min =",min_value,"\t max=",max_value)
    df_[feature_name+"_N"] = (df_[feature_name] - min_value) / (max_value - min_value)
    feature_name_div = max_value - min_value
    feature_name_add = min_value
'''
def addPaddingofZero(df):
    df["actor_X"] = 0.0
    df["actor_Y"] = 0.0
    df["critic_Value"] = 0.0


def normalize(df_,features, devideBy):
  print(features)
  for feature_name in features: 
    df_[feature_name+"_N"] = df_[feature_name]/devideBy

def dNormalize(df_,features, devideBy):
  print(features)
  for feature_name in features: 
    df_[feature_name+"_N"] = df_[feature_name]*devideBy
 

def removeOneHotColumns(properties):
      properties.remove('F1')
      properties.remove('F2')
      properties.remove('F3')
      properties.remove('F4')
      properties.remove('F5')
      properties.remove('F6')
      properties.remove('F7')
      properties.remove('F8')
      properties.remove('F9')
      properties.remove('F10')     
      return properties

def getConcatinateLayerOutput(test_nn, model):
    concat_func = K.function([model.layers[0].input],
                                [model.get_layer('concatenate').output])
    concat_output = concat_func([test_nn])
    #print (concat_output)
    return concat_output  

def getAccuracyLayerOutput(concat_output, model):
    accuracy_func = K.function([model.get_layer('dense_6').input],
                                [model.get_layer('dense_8').output])
    accuracy_output = accuracy_func([concat_output])
    print (accuracy_output)
    return accuracy_output  

def getGradientFromInitialInputVector(concat_output, model):
    grads = K.gradients(model.get_layer('dense_8').output, model.get_layer('concatenate').output)[0]
    grad_func = K.function([model.get_layer('concatenate').output], [grads])
    grads_value = grad_func([concat_output])
    print (grads_value)
    return grads_value



def update_xy_in_grad_direction(grads_value,concat_output):
    grad_array = np.asarray(grads_value)
    grad_x = grad_array.item(0)
    grad_y = grad_array.item(1)
    print (grad_x)
    print (grad_y)
    alpha = 0.001
    # get the x & y points from concat_output
    concat_output_array = np.asarray(concat_output)
    x_value = concat_output_array.item(0)
    y_value = concat_output_array.item(1)
    emb_x_repeat = x_value + alpha * grad_x
    emb_y_repeat = y_value + alpha * grad_y
    print(emb_x_repeat) 
    print(emb_y_repeat) 
    print("concat_output ")
    print(concat_output) 
    print(concat_output[0][0][0]) 
    print(concat_output[0][0][1]) 
    concat_output[0][0][0] = emb_x_repeat
    concat_output[0][0][1] = emb_y_repeat
    print(concat_output[0][0][0]) 
    print(concat_output[0][0][1]) 
    return emb_x_repeat,emb_y_repeat,concat_output

def getAction_probs(concat_output, model):
    print ('model.get_layer(dense_9).input.type:', type(model.get_layer('dense_9').input) )
    action_func = K.function([model.get_layer('dense_9').input],
                                    [model.get_layer('actor_output').output])
    action_probs = action_func(concat_output)
    print ("action_probs")
    print (action_probs)
    return action_probs

def getCritic_probs(concat_output, model):
    print ('model.get_layer(dense_10).input.type:', type(model.get_layer('dense_10').input) )
    critic_func = K.function([model.get_layer('dense_10').input],
                                    [model.get_layer('critic_output').output])
    critic_value = critic_func(concat_output)
    print ("critic_value")
    print (critic_value)
    return critic_value

# function to calculate next step for RL network
#   next_state = state + alpha * action
#   reward  = diff between the accuracy of (next state and state)
#   done = when is it acceeds 20 steps ([-0.07533027,  0.07512413],
def returnStepValues(action, alpha, state, prev_accuracy, steps_per_episode, max_steps_per_episode, model, done):

    step = action * alpha
    
    arrayChunk = np.split(state,[1,2])
    x_value =  arrayChunk[0]
    y_value = arrayChunk[1]
    oneHot = arrayChunk[2]
    xy_value = np.append(x_value, y_value)
    xy_valueT = tf.convert_to_tensor(xy_value)
    next_state = tf.add(xy_valueT, step) 
    oneHotT = tf.convert_to_tensor(oneHot)
    # Put back oneHotVector
    # Get np array from tensor append the onehot and then get the tensor back
    Combined_arr_T = tf.concat([next_state, oneHotT], axis = 0)
    Combined_arr = tf.keras.backend.get_value(Combined_arr_T)
    var_sizes = [np.product(list(map(int, v.get_shape())))*v.dtype.size
                for key in Combined_arr_T.graph.get_all_collection_keys() for v in Combined_arr_T.graph.get_collection_ref(key)]
    total = sum(var_sizes)/(1024**2)
    #for v in Combined_arr_T.all_variables():
        #vars += np.prod(v.get_shape().as_list())
        # vars contains the sum of the product of the dimensions of all the variables in your graph.
    file_episode_values.write(">>>> sum of the product of the dimensions of all the variables in Combined_arr_T graph: - " + str(total)+ "MB \n")	

    updated_Combined_arr = np.array([list(Combined_arr)])
    list_combined = []
    list_combined.append(updated_Combined_arr)
    #accuracy_state = getAccuracyLayerOutput(concat_output, model)
    #prev_accuracy = (accuracy_state[0][0][0]).item()
    #accuracy_state = getAccuracyLayerOutput(next_state,oneHotT  , model)
    t_accuracy_start = round(time.time() * 1000)
    t_accuracy_start_ps = round(time.process_time() * 1000)
    accuracy_next_state = getAccuracyLayerOutput(list_combined, model)
    
    t_accuracy_end = round(time.time() * 1000)
    t_accuracy_end_ps = round(time.process_time() * 1000)
    t_accuracy_total = t_accuracy_end - t_accuracy_start 
    t_accuracy_total_ps = t_accuracy_end_ps - t_accuracy_start_ps
    file_episode_values.write(">>>> Total Time to get accuracy_next_state in episode: - " + str(s) + "iteration - " + str(steps_per_episode) + " is " + str(t_accuracy_total)+ "\n")
    file_episode_values.write(">>>>>>>>>> Total Time to get accuracy_next_state_ps in episode: - " + str(s) + "iteration - " + str(steps_per_episode) + " is " + str(t_accuracy_total_ps)+ "\n")
    next_accuracy = (accuracy_next_state[0][0][0]).item()
    
    reward  =  next_accuracy - prev_accuracy 
    if steps_per_episode == max_steps_per_episode:
        done = True
    steps_per_episode =+ steps_per_episode + 1
    return Combined_arr,next_accuracy, reward, done, steps_per_episode


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
    #  We also need a terminal memory which will be of type boolean 
    self.done_memory = np.zeros((maxsize,), dtype= np.bool)

  # We need a function to store transitions where transition is 
  #   state,action,reward, new_state(Next_state) and terminal flag (done)
  def storexp(self, state, next_state, action, done, reward):
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
    return states, next_states, rewards, actions, dones



# Actor  network class
class Actor(tf.keras.Model):
    def __init__(self):
        super(Actor, self).__init__()    
        self.h1 = Dense(10, activation='relu',  name='h1')
        self.actor_output = Dense(2, activation='tanh', name='actor_output')

    #  We have call function for Actor for forward propagation operation.
    def call(self, predict_layer):
        x = self.h1(predict_layer)
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
        self.state_h1 = Dense(24, activation='relu', name='state_h1')
        self.critic_output = Dense(1, activation='tanh', name='critic_output')

    #  We have call function for Critic for forward propagation operation.
    def call(self, predict_layer, action):
        x = self.state_h1(tf.concat([predict_layer, action], axis=1))
        x = self.critic_output(x)
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
    self.batch_size = 64
    # number of our actions
    self.n_actions = 2  # As actor gives two dimension x,y output.
    # Adam optimizer for actor  
    self.a_opt = tf.keras.optimizers.Adam(0.001)
    # self.actor_target = tf.keras.optimizers.Adam(.001)
    self.c_opt1 = tf.keras.optimizers.Adam(0.002)
    self.c_opt2 = tf.keras.optimizers.Adam(0.002)
    
    # We dont need to be doing any gradient decent on 
    #    both actor and critic target networks. 
    #    We will be doing only soft network update on these target networks.
    #    In learning fucntion we dont call an update for the loss function for these.

    # self.actor_target = tf.keras.optimizers.Adam(.001)
    # self.critic_target = tf.keras.optimizers.Adam(.002)


    # maxsize for ReplyBuffer is defaulted to 1,000,000 # a million
    # input dimensions is env.observation_space.shape and actions is env.action_space.high
    #  env.observation_space.shape is state of the network 
    observation_space_shape = (12,)  # Ask Huber - assuming we have only two dimensional space
    self.memory = RBuffer(10000,  observation_space_shape, 2)
    self.trainstep = 0
    self.replace = 5
    # We need gamma - a discount factor for update equation
    self.gamma = 0.99

    # max/min actions for our environment 
    # Huber What are min/max (-1/+1) ?
    # Ask Huber - what are they in our case
    self.min_action = -1.0
    self.max_action = 1.0

    self.actor_update_steps = 2
    self.warmup = 200
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
      state = list(state ) 
      state = tf.convert_to_tensor([state], dtype=tf.float32)
      # now we pass state to the actor network to get the actions out.
      actions = self.actor_main(state)
      # For training, we added noise in action and for testing, we will not add any noise.
      if not evaluate:  # if we are training then we want to get some random normal noise
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
  def savexp(self,state, next_state, action, done, reward):
        self.memory.storexp(state, next_state, action, done, reward)

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
               file_new_policy_actions,file_next_state_target_value, file_target_values,file_episode_values,s,steps_per_episode):
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

      # Sample our memory after batch size is filled.
      states, next_states, rewards, actions, dones = self.memory.sample(self.batch_size)
      # convert states and new states, rewards and actions to tensor.
      states = tf.convert_to_tensor(states, dtype= tf.float32)
      next_states = tf.convert_to_tensor(next_states, dtype= tf.float32)
      rewards = tf.convert_to_tensor(rewards, dtype= tf.float32)
      actions = tf.convert_to_tensor(actions, dtype= tf.float32)
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
          t_Block_target_start = round(time.time() * 1000)
          t_Block_target_start_ps = round(time.process_time() * 1000)
          # target_actions is the target actor what are the things 
          # we should do for the new states
          # To avoid the looping issue it uses actor_target and keep actor_main stable
          target_actions = self.actor_target(next_states)
          print('############################### target_actions from actor_target ########################################')
          print(tf.keras.backend.get_value(target_actions))

          t_target_actions_start = round(time.time() * 1000)
          t_target_actions_start_ps = round(time.process_time() * 1000)
          target_actions += tf.clip_by_value(tf.random.normal(shape=[*np.shape(target_actions)], mean=0.0, stddev=0.2), -0.5, 0.5)
          target_actions = self.max_action * (tf.clip_by_value(target_actions, self.min_action, self.max_action))
          t_target_actions_end = round(time.time() * 1000)
          tt_target_actions_start = t_target_actions_end - t_target_actions_start
          t_target_actions_end_ps = round(time.process_time() * 1000)
          tt_target_actions_start_ps = t_target_actions_end_ps - t_target_actions_start_ps 
          file_episode_values.write(">>>> Total Time to get target_actions in episode: - " + str(s) + "iteration - " + str(steps_per_episode) + " is " + str(tt_target_actions_start)+ "\n")
          file_episode_values.write(">>>>>>>> Total Time to get target_actions_ps in episode: - " + str(s) + "iteration - " + str(steps_per_episode) + " is " + str(tt_target_actions_start_ps)+ "\n")
          
          print('############################### target_actions after clip and updates ########################################')

          file_target_actions.write(np.array2string(tf.keras.backend.get_value(target_actions), precision=8, separator=','))
          #print(tf.keras.backend.get_value(target_actions)) 
          var_sizes = [np.product(list(map(int, v.get_shape())))*v.dtype.size
                for key in target_actions.graph.get_all_collection_keys() for v in target_actions.graph.get_collection_ref(key)]
          total = sum(var_sizes)/(1024**2)
          #for v in target_actions.all_variables():
            #vars += np.prod(v.get_shape().as_list())
            # vars contains the sum of the product of the dimensions of all the variables in your graph.
          file_episode_values.write(">>>> sum of the product of the dimensions of all the variables in target_actions graph: - " + str(total)+ "MB \n")	
            
          # critic value for new states is shown below as : target critic evaluation 
          # of the next states and target actions and squeeze along the first dimension.
          # It does the forward pass. 
          # The value of the succesor state for the best action

          t_target_next_state_values_start = round(time.time() * 1000)
          t_target_next_state_values_start_PS = round(time.process_time() * 1000)
          target_next_state_values = tf.squeeze(self.critic_target(next_states, target_actions), 1)
          print('############################### target_next_state_values ########################################')
          print(tf.keras.backend.get_value(target_next_state_values)) 
          file_target_next_state_values.write(np.array2string(tf.keras.backend.get_value(target_next_state_values), precision=8, separator=','))  
          target_next_state_values2 = tf.squeeze(self.critic_target2(next_states, target_actions), 1)
          t_target_next_state_values_end = round(time.time() * 1000)
          t_target_next_state_values_end_PS = round(time.process_time() * 1000)
          tt_target_next_state_values = t_target_next_state_values_end - t_target_next_state_values_start 
          tt_target_next_state_values_PS = t_target_next_state_values_end_PS - t_target_next_state_values_start_PS
          file_episode_values.write(">>>> Total Time to get target_next_state_values in episode: - " + str(s) + "iteration - " + str(steps_per_episode) + " is " + str(tt_target_next_state_values)+ "\n")
          file_episode_values.write(">>>>>>>>>>>>>>>>>>>> Total Time to get target_next_state_values_PS in episode: - " + str(s) + "iteration - " + str(steps_per_episode) + " is " + str(tt_target_next_state_values_PS)+ "\n")

          print('############################### target_next_state_values2 ########################################')
          #print(tf.keras.backend.get_value(target_next_state_values2))
 
          file_target_next_state_values2.write(np.array2string(tf.keras.backend.get_value(target_next_state_values2), precision=8, separator=','))
          var_sizes = [np.product(list(map(int, v.get_shape())))*v.dtype.size
                for key in target_next_state_values2.graph.get_all_collection_keys() for v in target_next_state_values2.graph.get_collection_ref(key)]
          total = sum(var_sizes)/(1024**2)
          #for v in target_next_state_values2.all_variables():
            #vars += np.prod(v.get_shape().as_list())
            # vars contains the sum of the product of the dimensions of all the variables in your graph.
          file_episode_values.write(">>>> sum of the product of the dimensions of all the variables in target_next_state_values2 graph: - " + str(total)+ " MB \n")	
           
          t_Block_target_end = round(time.time() * 1000)
          t_Block_target_end_ps = round(time.process_time() * 1000)
          t_Block_target_total = t_Block_target_end - t_Block_target_start 
          t_Block_target_total_ps = t_Block_target_end_ps - t_Block_target_start_ps
          file_episode_values.write(">>>> Total Time to get target For episode: " + str(s) + " iteration - " + str(steps_per_episode) + " is  - "+ str(t_Block_target_total) + "\n")
          file_episode_values.write(">>>>>>>>>>>>>> Total Time to get target_ps For episode: " + str(s) + " iteration - " + str(steps_per_episode) + " is  - "+ str(t_Block_target_total_ps) + "\n")

          # Our predicted (critic) values are the output of the main critic network which takes
          #  states and actions from the buffer sample.
          # critic value is the value of the current state's with respect to original state
          # and actions the agent actually took during the course of this episode.
          # This gives value of the action in a given state.
          t_Block_critic_start = round(time.time() * 1000)
          t_Block_critic_start_ps = round(time.process_time() * 1000)
          critic_value = tf.squeeze(self.critic_main(states, actions), 1)
          print('############################### critic_value ########################################')
          print(tf.keras.backend.get_value(critic_value)) 
          file_critic_value.write(np.array2string(tf.keras.backend.get_value(critic_value), precision=8, separator=','))  
          critic_value2 = tf.squeeze(self.critic_main2(states, actions), 1)
          t_Block_critic_squeeze = round(time.time() * 1000)
          t_Block_critic_squeeze_ps = round(time.process_time() * 1000)
          t_Block_critic_squeeze_total = t_Block_critic_squeeze - t_Block_critic_start 
          t_Block_critic_squeeze_total_ps = t_Block_critic_squeeze_ps - t_Block_critic_start_ps 
          file_episode_values.write(">>>> Total Time to get critic_squeeze For episode: " + str(s) + " iteration - " + str(steps_per_episode) + " is  - "+ str(t_Block_critic_squeeze_total) + "\n")
          file_episode_values.write(">>>>>>>>>>>>>>>>> Total Time to get critic_squeeze_PS For episode: " + str(s) + " iteration - " + str(steps_per_episode) + " is  - "+ str(t_Block_critic_squeeze_total_ps) + "\n")

          print('############################ critic_value2 ########################################')
          #print(tf.keras.backend.get_value(critic_value2))  
          file_critic_value2.write(np.array2string(tf.keras.backend.get_value(critic_value2), precision=8, separator=','))  
          var_sizes = [np.product(list(map(int, v.get_shape())))*v.dtype.size
                for key in critic_value2.graph.get_all_collection_keys() for v in critic_value2.graph.get_collection_ref(key)]
          total = sum(var_sizes)/(1024**2)
          #for v in critic_value2.all_variables():
            #vars += np.prod(v.get_shape().as_list())
            # vars contains the sum of the product of the dimensions of all the variables in your graph.
          file_episode_values.write(">>>> sum of the product of the dimensions of all the variables in critic_value2 graph: - " + str(total)+ " MB \n")	          
          next_state_target_value = tf.math.minimum(target_next_state_values, target_next_state_values2)
          print('############################ next_state_target_value ########################################')
          #print(tf.keras.backend.get_value(next_state_target_value))  
          file_next_state_target_value.write(np.array2string(tf.keras.backend.get_value(next_state_target_value), precision=8, separator=','))
          var_sizes = [np.product(list(map(int, v.get_shape())))*v.dtype.size
                for key in next_state_target_value.graph.get_all_collection_keys() for v in next_state_target_value.graph.get_collection_ref(key)]
          total = sum(var_sizes)/(1024**2)
          #for v in next_state_target_value.all_variables():
            #vars += np.prod(v.get_shape().as_list())
            # vars contains the sum of the product of the dimensions of all the variables in your graph.
          file_episode_values.write(">>>> sum of the product of the dimensions of all the variables in next_state_target_value graph: - " + str(total)+ " MB \n")	           

          # target for the terminal new state is just the reward for every other state
          # it is reward + discounted value of the resulting state according to the target critic network.
          # Then, we apply the Bellman equation to calculate target values 
          # (target_values = rewards + self.gamma * target_next_state_values * done)
          # If done (0) then there is no succesor state then the target value is the reward.
          target_values = rewards + self.gamma * next_state_target_value * dones
          print('############################ target_values = rewards + self.gamma * next_state_target_value * dones ########################################')
          #print(tf.keras.backend.get_value(target_values))    
          file_target_values.write(np.array2string(tf.keras.backend.get_value(target_values), precision=8, separator=','))         
          var_sizes = [np.product(list(map(int, v.get_shape())))*v.dtype.size
                for key in target_values.graph.get_all_collection_keys() for v in target_values.graph.get_collection_ref(key)]
          total = sum(var_sizes)/(1024**2)
          #for v in target_values.all_variables():
            #vars += np.prod(v.get_shape().as_list())
            # vars contains the sum of the product of the dimensions of all the variables in your graph.
          file_episode_values.write(">>>> sum of the product of the dimensions of all the variables in target_values graph: - " + str(total)+ " MB \n")	           
          
          # The loss function computed out of three network,actor_target, critic_target & critic_main
          # Critic loss is then calculated as MSE of target values and predicted values.
          # Boltzmann error is the diff (MSE) between the value of the action in the current state and
          # (Reward + gamma times max of all possible action of the value in the successor state )
          t_Block_critic_loss_start = round(time.time() * 1000)
          t_Block_critic_loss_start_ps = round(time.process_time() * 1000)
          critic_loss1 = tf.keras.losses.MSE(target_values, critic_value)
          critic_loss2 = tf.keras.losses.MSE(target_values, critic_value2)
          t_Block_critic_loss_end = round(time.time() * 1000)
          t_Block_critic_loss_end_ps = round(time.process_time() * 1000)
          t_Block_critic_loss_total = t_Block_critic_loss_start - t_Block_critic_loss_end 
          t_Block_critic_loss_total_ps = t_Block_critic_loss_start_ps - t_Block_critic_loss_end_ps 
          file_episode_values.write(">>>> Total Time to get critic_loss For episode: " + str(s) + " iteration - " + str(steps_per_episode) + " is  - "+ str(t_Block_critic_loss_total) + "\n")
          file_episode_values.write(">>>>>>>>> Total Time to get critic_loss_ps For episode: " + str(s) + " iteration - " + str(steps_per_episode) + " is  - "+ str(t_Block_critic_loss_total_ps) + "\n")

          print('############################ critic_loss1 ########################################')
          #print(tf.keras.backend.get_value(critic_loss1)) 
          file_critic_loss.write(np.array2string(tf.keras.backend.get_value(critic_loss1), precision=8, separator=',')+ "\n")  
          var_sizes = [np.product(list(map(int, v.get_shape())))*v.dtype.size
                for key in critic_loss1.graph.get_all_collection_keys() for v in critic_loss1.graph.get_collection_ref(key)]
          total = sum(var_sizes)/(1024**2)
          #for v in critic_loss1.all_variables():
            #vars += np.prod(v.get_shape().as_list())
            # vars contains the sum of the product of the dimensions of all the variables in your graph.
          file_episode_values.write(">>>> sum of the product of the dimensions of all the variables in critic_loss1 graph: - " + str(total)+ " MB \n")	           
 
          print('############################ critic_loss2 ########################################')
          #print(tf.keras.backend.get_value(critic_loss2)) 
          file_critic_loss2.write(np.array2string(tf.keras.backend.get_value(critic_loss2), precision=8, separator=',')+ "\n")  
          var_sizes = [np.product(list(map(int, v.get_shape())))*v.dtype.size
                for key in critic_loss2.graph.get_all_collection_keys() for v in critic_loss2.graph.get_collection_ref(key)]
          total = sum(var_sizes)/(1024**2)
          #for v in critic_loss2.all_variables():
            #vars += np.prod(v.get_shape().as_list())
            # vars contains the sum of the product of the dimensions of all the variables in your graph.
          file_episode_values.write(">>>> sum of the product of the dimensions of all the variables in critic_loss2 graph: - " + str(total)+ " MB \n")	           
          
          t_Block_critic_end = round(time.time() * 1000)
          t_Block_critic_end_ps = round(time.process_time() * 1000)
          t_Block_critic_total = t_Block_critic_end - t_Block_critic_start 
          t_Block_critic_total_ps = t_Block_critic_end_ps - t_Block_critic_start_ps 
          file_episode_values.write(">>>> Total Time to get critic For episode: " + str(s) + " iteration - " + str(steps_per_episode) + " is  - "+ str(t_Block_critic_total) + "\n")
          file_episode_values.write(">>>>>>>>>>>>> Total Time to get critic_ps For episode: " + str(s) + " iteration - " + str(steps_per_episode) + " is  - "+ str(t_Block_critic_total_ps) + "\n")

      # for critic loss calculate gradients and apply the gradients 
      # Compute grad with respect to critic_main 
      t_Block_grads12_start = round(time.time() * 1000)
      t_Block_grads12_start_ps = round(time.process_time() * 1000)
      grads1 = tape1.gradient(critic_loss1, self.critic_main.trainable_variables)
      grads2 = tape2.gradient(critic_loss2, self.critic_main2.trainable_variables)

      # Apply our gradients on same critic_main trainable_variables 
      # All the weights of the three alyers in the  main critic network
      # Gradient involves all the weights of all three networks and we 
      # apply only to the main critic network weights of all layers only
      self.c_opt1.apply_gradients(zip(grads1, self.critic_main.trainable_variables))
      self.c_opt2.apply_gradients(zip(grads2, self.critic_main2.trainable_variables))

      self.trainstep +=1
      t_Block_grads12_end = round(time.time() * 1000)
      t_Block_grads12_total = t_Block_grads12_end - t_Block_grads12_start
      t_Block_grads12_end_ps = round(time.process_time() * 1000)
      t_Block_grads12_total_ps = t_Block_grads12_end_ps - t_Block_grads12_start_ps
      file_episode_values.write(">>>> Total Time to get grads12 For episode: " + str(s) + " iteration - " + str(steps_per_episode) + " is  - "+ str(t_Block_grads12_total) + "\n")
      file_episode_values.write(">>>>>>>>>>> Total Time in ps to get grads12 For episode: " + str(s) + " iteration - " + str(steps_per_episode) + " is  - "+ str(t_Block_grads12_total_ps) + "\n")
      # Ask Huber
      if self.trainstep % self.actor_update_steps == 0:
                
          with tf.GradientTape() as tape3:
            # These are the actions according to actor based upon its current 
            # set of weights. Not based upon the weights it had at the time 
            # whatever the memory we stored in a agent's memory
            new_policy_actions = self.actor_main(states)
            t_Block_new_policy_actions_end = round(time.time() * 1000)
            t_Block_new_policy_actions_end_ps = round(time.process_time() * 1000)
            t_Block_new_policy_actions_Total = t_Block_new_policy_actions_end - t_Block_grads12_end
            t_Block_new_policy_actions_Total_ps = t_Block_new_policy_actions_end_ps - t_Block_grads12_end_ps
            file_episode_values.write(">>>> Total Time to get new_policy_actions For episode: " + str(s) + " iteration - " + str(steps_per_episode) + " is  - "+ str(t_Block_new_policy_actions_Total) + "\n")
            file_episode_values.write(">>>>>>>>>> Total Time to get new_policy_actions_ps For episode: " + str(s) + " iteration - " + str(steps_per_episode) + " is  - "+ str(t_Block_new_policy_actions_Total_ps) + "\n")

            print('############################ new_policy_actions ########################################')
            #print(tf.keras.backend.get_value(new_policy_actions))
            file_new_policy_actions.write(np.array2string(tf.keras.backend.get_value(new_policy_actions), precision=8, separator=','))
            var_sizes = [np.product(list(map(int, v.get_shape())))*v.dtype.size
                for key in new_policy_actions.graph.get_all_collection_keys() for v in new_policy_actions.graph.get_collection_ref(key)]
            total = sum(var_sizes)/(1024**2)
            #for v in new_policy_actions.all_variables():
                #vars += np.prod(v.get_shape().as_list())
                # vars contains the sum of the product of the dimensions of all the variables in your graph.
            file_episode_values.write(">>>> sum of the product of the dimensions of all the variables in new_policy_actions graph: - " + str(total)+ " MB \n")	           
  
            # It is negative ('-') as we are doing gradient ascent.
            # As in policy gradient methods we dont want to use decent 
            # because that will minimize the total score over time. 
            # Rather we would like to maximize the total score over time.
            # Gradient ascent is just negative of gradient decent.
            # Actor loss is calculated as negative of critic main values with 
            # inputs as the main actor predicted actions.
            t_Block_actor_loss_start = round(time.time() * 1000)
            t_Block_actor_loss_start_ps = round(time.process_time() * 1000)
            actor_loss = -self.critic_main(states, new_policy_actions) 
            # Then our loss is reduce mean of that actor loss.
            actor_loss = tf.math.reduce_mean(actor_loss)
            t_Block_actor_loss_end = round(time.time() * 1000)
            t_Block_actor_loss_end_ps = round(time.process_time() * 1000)
            t_Block_actor_loss_total = t_Block_actor_loss_end - t_Block_actor_loss_start 
            t_Block_actor_loss_total_ps = t_Block_actor_loss_end_ps -  t_Block_actor_loss_start_ps
            file_episode_values.write(">>>> Total Time to get inner actor_loss For episode: " + str(s) + " iteration - " + str(steps_per_episode) + " is  - "+ str(t_Block_actor_loss_total) + "\n")
            file_episode_values.write(">>>>>>>>>> Total Time to get inner actor_loss_ps For episode: " + str(s) + " iteration - " + str(steps_per_episode) + " is  - "+ str(t_Block_actor_loss_total_ps) + "\n")

            print('############################ actor_loss ########################################')
            #print(tf.keras.backend.get_value(actor_loss))   
            file_actor_loss.write(np.array2string(tf.keras.backend.get_value(actor_loss), precision=8, separator=',') + "\n")            
            var_sizes = [np.product(list(map(int, v.get_shape())))*v.dtype.size
                for key in actor_loss.graph.get_all_collection_keys() for v in actor_loss.graph.get_collection_ref(key)]
            total = sum(var_sizes)/(1024**2)
            #for v in actor_loss.all_variables():
                #vars += np.prod(v.get_shape().as_list())
                # vars contains the sum of the product of the dimensions of all the variables in your graph.
            file_episode_values.write(">>>> sum of the product of the dimensions of all the variables in actor_loss graph: - " + str(total)+ " MB \n")	           
          
            # In the paper they applied the chain rule gradient of critic network with actor network
            # This is how we get the gradient of the critic loss with respect to Meu (µ) parameter 
            # by taking this actor loss which is proportional to the output of the critic network
            # and is coupled. The gradient is non zero because it has this dependency on the output of our 
            # actor networks. Dependence bacause of non-zero gradient comes from the fact that we are taking actions
            #  with respect to actor network, which is calculated according to theatas (Ɵ super µ)
            #  That can effect from here to the critic network.  That's what allows to take the gradient of the output of the 
            # critic network with respect to the variables of the actor network that's how we get coupling.
            t_Block_actor_loss_end = round(time.time() * 1000)
            t_Block_actor_loss_end_ps = round(time.process_time() * 1000)
            t_Block_actor_loss_Total =   t_Block_actor_loss_end - t_Block_new_policy_actions_end
            t_Block_actor_loss_Total_ps = t_Block_actor_loss_end_ps - t_Block_new_policy_actions_end_ps 
            file_episode_values.write(">>>> Total Time to get actor_loss For episode: " + str(s) + " iteration - " + str(steps_per_episode) + " is  - "+ str(t_Block_actor_loss_Total) + "\n")
            file_episode_values.write(">>>>>>>>> Total Time to get actor_loss_ps For episode: " + str(s) + " iteration - " + str(steps_per_episode) + " is  - "+ str(t_Block_actor_loss_Total_ps) + "\n")

          # Since actor_loss involves actor_main & critic_main
          grads3 = tape3.gradient(actor_loss, self.actor_main.trainable_variables)
          # Apply the gradients to actor_main trainable variables.
          self.a_opt.apply_gradients(zip(grads3, self.actor_main.trainable_variables))
          t_Block_grads3_end = round(time.time() * 1000)
          t_Block_grads3_end_ps = round(time.process_time() * 1000)
          t_Block_grads3_Total = t_Block_grads3_end - t_Block_actor_loss_end
          t_Block_grads3_Total_ps = t_Block_grads3_end_ps - t_Block_actor_loss_end_ps
          file_episode_values.write(">>>> Total Time to get grads3 For episode: " + str(s) + " iteration - " + str(steps_per_episode) + " is  - " +  str(t_Block_grads3_Total_ps)    + "\n")
          file_episode_values.write(">>>>>>>>>>> Total Time to get grads3_ps For episode: " + str(s) + " iteration - " + str(steps_per_episode) + " is  - " +  str(t_Block_grads3_Total)    + "\n")

      #if self.trainstep % self.replace == 0:
      # Perform soft update on our target network. We use default value of tau = 0.005
      # we update our target networks with a tau of 0.005.
      t_ut_start = round(time.time() * 1000)
      t_ut_start_ps = round(time.process_time() * 1000)
      self.update_target()  
      t_ut_end = round(time.time() * 1000)
      t_ut_end_ps = round(time.process_time() * 1000)
      tt_ut = t_ut_end - t_ut_start 
      tt_ut_ps = t_ut_end_ps - t_ut_start_ps 
      file_episode_values.write(">>>> Total Time to get update_target For episode: " +str(s)  + " iteration - " + str(steps_per_episode) + " is  - " + str(tt_ut)+ "\n")
      file_episode_values.write(">>>>>>>>>>>> Total Time to get update_target_ps For episode: " +str(s)  + " iteration - " + str(steps_per_episode) + " is  - " + str(tt_ut_ps)+ "\n")
      # current date and time in Eastcost
      #date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
      #file_episode_values.write("date and time after training completed :" +date_time+ "\n")	 
      #file_episode_values.write(" For episode:"+str(s)+ "\n")	 
      return target_actions,target_next_state_values,target_next_state_values2,critic_value,critic_value2,next_state_target_value,target_values,critic_loss1,critic_loss2,new_policy_actions,actor_loss   




#
#        CODE FOR NETWORK ONE without RL
#   
# Dense Layers nodes for encoder and decoder

latent_dim = 10
devideBy = 0.0
current_dir  = "/home/bvadhera/huber/"
# load The TrainingDataSet File
inputNN_Architectures_DataSet = current_dir+"combined_nn_architectures_OneHot_orig.csv"
print (inputNN_Architectures_DataSet)

writeFile = current_dir + "results_" + "combined_nn_architectures_OneHot.csv"

myFile = open(writeFile, 'w')

##  TODO - CHange the headings as per the results
header = ['training_loss','val_loss', 
                      'val_loss', 'decoder_accuracy', 'mean_sqe_pred' , 
                      'val_decoder_accuracy', 'val_mean_sqe_pred',
                      'test_loss', 'test_decoder_accuracy','test_mean_sqe_pred','test_acc'] 

writer = csv.DictWriter(myFile, fieldnames=header) 
writer.writeheader()
print("writeFile")
print(writeFile)

# Open a csv reader called DictReader
# iterate over each line as a ordered dictionary and print only few column by column name
df = pd.read_csv(inputNN_Architectures_DataSet)
print(df.head())
normalizeProp = list(df.columns.values)
# remove 'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'
removeOneHotColumns(normalizeProp)
#Also remove 'test_acc' so that we dont normalize the accuracy
normalizeProp.remove('test_acc')
print("normalizeProp")
print(normalizeProp)
devideBy = getMaxNumOfNodesInDataSet(df) + 2
#Normalize the data for num_layers, num_node1, num_node2 only
normalize(df,normalizeProp,devideBy)
print("df.head() after normalized data")
print(df.head())
origProperties = list(df.columns.values)
trueProperties = list(df.columns.values)
# Now save two dataframes one for final test which is y (num_layers_N  num_node1_N  num_node2_N, test_acc)
# and another for X (num_layers_N, num_node1_N, num_node2_N,,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10)

# remove properties 'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'
#  and 'num_layers', 'num_node1', 'num_node2'
removeOneHotColumns(trueProperties)
trueProperties.remove('num_layers')
trueProperties.remove('num_node1')
trueProperties.remove('num_node2')
#trueProperties.remove('test_acc')
print("trueProperties")
print(trueProperties)
y = df[trueProperties]
# add three columns of 0 to pad 
# Bhanu Add 000 

# add three columns of 0 to pad 
# Bhanu Add 000 
addPaddingofZero(y)


print("y_true.head() after normalized data")
print(y.head())
xOrigProperties = list(df.columns.values)
xOrigProperties.remove('num_layers')
xOrigProperties.remove('num_node1')
xOrigProperties.remove('num_node2')
xOrigProperties.remove('test_acc') 
xProperties = xOrigProperties
print("xProperties")
print(xProperties)
X = df[xProperties]
print("X.head() with only num_layers_N  num_node1_N  num_node2_N and one hot vector")
print(X.head())

# Divide the data into test and train sets
a = random.randint(1,50)
print ("random_state")
print (a)
X_train = X
y_train = y

print ('X_train.shape:', X_train.shape)  # shape 
print ('y_train.shape:',y_train.shape)   # shape 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=a)
print("X_train.head() ")
print(X_train.head())

print("y_train.head() ")
print(y_train.head())
print("y_test.head() ")
print(y_test.head())

print("X_test.head() ")
print(X_test.head())
print("X_test type ")
print(type(X_test) )
print ('X_test.shape:', X_test.shape)  # shape 


print ('X_train.shape:', X_train.shape)  # shape 
print ('y_train.shape:',y_train.shape)   # shape 

print("The type of x_train is : ",type(X_train))
# Get Batch size for X_train
batchSize = 50
print ("batchSize")
print (batchSize)   

# Create a joint NN with split input layer(s))
input_layer = Input(shape=(13,))
print("input_layer.shape")
print(input_layer.shape)
print("type(input_layer)")
print(type(input_layer) )
split = Lambda( lambda x: tf.split(x,[10,3],axis=1))(input_layer)
#split1 = Lambda( lambda x: tf.split(x,[11,2],axis=1))(input_layer)
print ('split[0].shape:', split[0].shape)  # shape one hot vector
print ('split[1].shape:',split[1].shape)   # shape  num_layers_N  num_node1_N  num_node2_N

# to use them as parellel layers

#+++++++++++++++++++++++++++++++++++++
# Building TRACK - 1
#++++++++++++++++++++++++++++++++
# nn_layer goes int encoder  
#Hidden Layers
encoder_layer_1 = Dense(latent_dim, activation=tf.nn.leaky_relu)((split[1]))
print ('encoder_layer_1.shape:', encoder_layer_1.shape)
encoder_layer_2 = Dense(latent_dim, activation=tf.nn.leaky_relu)(encoder_layer_1) 
#output embedding (x,y) Layer
embedding_layer = Dense(int(2), activation=tf.nn.leaky_relu)(encoder_layer_2) 

print ('embedding_layer.shape:', embedding_layer.shape)

#++++++++++++++++++++++++++++++++
# embedding_layer goes int decoder  
#Hidden Layers
decoder_layer_1 = Dense(latent_dim, activation=tf.nn.leaky_relu)(embedding_layer)
decoder_layer_2 = Dense(latent_dim, activation=tf.nn.leaky_relu)(decoder_layer_1) 
#output decoded (x,y) Layer
decoded_layer = Dense(int(3), activation=tf.nn.leaky_relu)(decoder_layer_2)
print ('decoded_layer.shape:', decoded_layer.shape)

#++++++++++++++++++++++++++++++++
#predict_layer = Concatenate()([split[0],decoded_layer])
#+++++++++++++++++++++++++++++++++++++
# Building  TRACK - 2
#++++++++++++++++++++++++++++++++
#oneHot_layer = Dense(5)(split[0]) # just need  one hot vector
#print ('oneHot_layer.shape:', oneHot_layer.shape)
#Merge oneHot_layer  back with embedding_layer
predict_layer = Concatenate()([embedding_layer,split[0]])

# Since split[0] is  num_layers_N  num_node1_N  num_node2_N
# we need only num_node1_N  num_node2_N
#splitAgain = Lambda( lambda x: tf.split(x,[1,2],axis=1))(split[1])
#print ('splitAgain[0].shape:', splitAgain[0].shape)  # shape num_layers_N  (1)
#print ('splitAgain[1].shape:',splitAgain[1].shape)   # shape  num_node1_N  num_node2_N (2)
#predict_layer = Concatenate()([splitAgain[1],split[0]])

#predict_layer = Concatenate()([split1[1],split[0]])
print ('predict_layer.shape:', predict_layer.shape)


#Hidden Layers
hiddenP_layer_1 = Dense(int(10), activation=tf.nn.leaky_relu)(predict_layer)
print ('hiddenP_layer_1.shape:', hiddenP_layer_1.shape)
hiddenP_layer_2 = Dense(int(5), activation=tf.nn.leaky_relu)(hiddenP_layer_1)
print ('hiddenP_layer_2.shape:', hiddenP_layer_2.shape)
#Call Model to get predicted Accuracy as output
accuracy_layer = Dense(int(1), activation='sigmoid')(hiddenP_layer_2)
print ('accuracy_layer.shape:', accuracy_layer.shape)


#out put of prediction accuracy_layer


#+++++++++++++++++++++++++++++++++++++
# Building Final  JOINT TRACK  
#++++++++++++++++++++++++++++++++
#Merge predict_layer (predicted Accuracy) back with decoded_layer
predict_output = Concatenate(name='network_with_accuracy')([accuracy_layer,decoded_layer])
print ('predict_output.shape:', predict_output.shape)

###### Huber Removed here RL TRack


#Call Model to get Final Accuracy as output with custom MSE loss function
##Defining the model by specifying the input and output layers
# model = Model(inputs=input_layer, outputs=[predict_output,actor_output, critic_output])
#Error Dr Huber ???
###### Huber Removed here RL  
model = Model(inputs=input_layer, outputs=[predict_output])
for idx in range(len(model.layers)):
  print(model.get_layer(index = idx).name)
print(model.summary())
# Instantiate an accuracy metric. custom accuracy  type
#accuracyType = tf.keras.metrics.RootMeanSquaredError()
# Instantiate an optimizer.
learning_rate = 0.01
optimizer = tf.keras.optimizers.Adam(learning_rate)
filepath=current_dir +"acc." #modelFile+"_"+"acc."
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max') #monitor='val_tpr',
callbacks_list_NN = [checkpoint]
#First, call compile to configure the optimizer, loss, and metrics to monitor. 
###### Huber Removed here RL  
model.compile(optimizer=optimizer, loss={'network_with_accuracy': custom_loss},
                     metrics={'network_with_accuracy': [decoder_accuracy , mean_sqe_pred]}) #, run_eagerly=True)
# Dr Huber check
#model.compile(optimizer=optimizer, loss={'network_with_accuracy': custom_loss, 'actor_output': 'huber_loss', 'critic_output': 'huber_loss'} , 
#                        metrics={'network_with_accuracy':[decoder_accuracy , mean_sqe_pred],'actor_output': mean_sqe_pred, 'critic_output': mean_sqe_pred} )
#history = model.fit(X_train, y_train, validation_split=0.1,  callbacks=callbacks_list_NN, verbose=2, epochs=1, batch_size=batchSize)
print (type(y_train))

###### Huber Removed here RL 
history = model.fit(X_train, {'network_with_accuracy': y_train[['test_acc', 'num_layers_N', 'num_node1_N', 'num_node2_N']] }, validation_split=0.1,  
                                                                    callbacks=callbacks_list_NN, verbose=2, epochs=1, batch_size=batchSize)
print (history.history)

### Try Bhanu
#sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph())
#session = tf.compat.v1.keras.backend.set_session(sess)
#graph = tf.compat.v1.get_default_graph() 
#test = tf.constant([1,2,3])
#print(sess.run(test))
#model.save(current_dir)



training_loss = history.history["loss"]
###### Huber Removed here RL 
#network_with_accuracy_loss  = history.history["network_with_accuracy_loss"]
val_loss = history.history["val_loss"]
#val_network_with_accuracy_loss = history.history["val_network_with_accuracy_loss"]
decoder_accuracy_print = history.history["decoder_accuracy"]
mean_sqe_pred_print = history.history["mean_sqe_pred"]
val_decoder_accuracy = history.history["val_decoder_accuracy"]
val_mean_sqe_pred = history.history["val_mean_sqe_pred"] 
print (X_test)
#  I need one NN from X_test
# For now i take ist  NN  where F5 = 1
print ((X_test.loc[X_test['F5'] == 1]).iloc[[0]])
test_nn = (X_test.loc[X_test['F5'] == 1]).iloc[[0]]
print("test_nn.head() ")
print(test_nn.head())
print("test_nn type ")
print(type(test_nn)) 
print ('test_nn.shape:', test_nn.shape)  # shape 


# Generate predictions for samples
#predictions, actor, critic = model.predict(test_nn)
# Error Dr Huber
predictions  = model.predict(test_nn)
print("predictions ========")
print(predictions[0])
#print(actor)
#print(critic)
#Take the accurecy value out of test_nn
drived_acc = predictions[0]
print (drived_acc)
print(model.summary())

#--------------------
#Get the test data for gradient calculations

testProperties = list(test_nn.columns.values)
print (testProperties)
testProperties.remove('num_layers_N')
testProperties.remove('num_node1_N')
testProperties.remove('num_node2_N')
df_oneHot = test_nn[testProperties]
print("df_oneHot.head()")
print(df_oneHot.head())
df_array = df_oneHot.to_numpy()
f1 = str(df_array.item(0))
f2 = str(df_array.item(1))
f3 = str(df_array.item(2))
f4 = str(df_array.item(3))
f5 = str(df_array.item(4))
f6 = str(df_array.item(5))
f7 = str(df_array.item(6))
f8 = str(df_array.item(7))
f9 = str(df_array.item(8))
f10 = str(df_array.item(9))

print(type(f1))

testHot =  f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10

print("testHot as string ")
print(testHot) 
print(type(testHot))
num1 = 50
num2 = 10
#NN Architecture File Name 
writeGradFile = current_dir + "outputData"+"/nn_grad_architectures_" +  testHot + ".csv"
print(writeGradFile)
myGradFile = open(writeGradFile, 'w')
with myGradFile:    
    header = ['num_layers', 'num_node1', 'num_node2','F1','F2','F3',
             'F4','F5','F6','F7','F8','F9','F10','test_acc']
    gradWriter = csv.DictWriter(myGradFile, fieldnames=header)    
    gradWriter.writeheader()
    #For Loop for 50 NN
    for index in range(num1):
        concat_output = getConcatinateLayerOutput(test_nn, model)
        print("concat_output type ")
        print(type(concat_output)) 
        print("concat_output ")
        print(concat_output) 
        #do 10 iterations in gradient direction
        for index1 in range(num2):
            grads_value = getGradientFromInitialInputVector(concat_output, model)
            emb_x_repeat,emb_y_repeat,concat_output = update_xy_in_grad_direction(grads_value,concat_output)

    #-----------------------
        list_x_y_repeat = [emb_x_repeat, emb_y_repeat]
        print (type(list_x_y_repeat))
    
        x_y_repeat_T = K.constant(np.asarray(list_x_y_repeat), shape=(1,2))
        print (x_y_repeat_T)
        print ('x_y_repeat_T.shape:', x_y_repeat_T.shape) 
        print ('x_y_repeat_T.type:', type(x_y_repeat_T)) 
        #Take enbedding value of that NN and add alpa (0.01) times the gradient 
        #send this through the decoder and get the value of network 
        print ('model.get_layer(dense_2).output.shape:', model.get_layer('dense_2').output.shape) 
        print ('model.get_layer(dense_2).output.type:', type(model.get_layer('dense_2').output) )
        decod_func = K.function([model.get_layer('dense_2').output],
                                    [model.get_layer('dense_5').output])
        decod_output = decod_func([[x_y_repeat_T]])
        print ("decod_output")
        print (decod_output)
        print ("TYPE OF decod_output")
        print (type(decod_output))
        # get the x & y points from concat_output
        decod_output_array = np.asarray(decod_output)
        n_layers = decod_output_array.item(0)
        num_node1 = decod_output_array.item(1)
        num_node2 = decod_output_array.item(2)
        #attach new NN to the OnHot
        if index != 0:
            df_oneHot = test_nn[testProperties]
        print("df_oneHot.head()")
        print(df_oneHot.head())
        df_oneHot.insert(10, 'num_layers_N', [np.asarray(decod_output).item(0)])
        df_oneHot.insert(11, 'num_node1_N', [np.asarray(decod_output).item(1)])
        df_oneHot.insert(12, 'num_node2_N', [np.asarray(decod_output).item(2)])
        test_nn = df_oneHot 
        # get prediction for this decoded values
        # Generate predictions for samples
        print("df_oneHot.head()")
        print(df_oneHot.head())
        print(test_nn.head())
        predictions = model.predict(test_nn)
        # Just get only Network_Accuracy part of prediction.
        print("predictions ========")
        print(predictions)
        nn_predictions = predictions
        print("nn_predictions ========")
        print(nn_predictions)
        #Take the accurecy value out of test_nn
        drived_acc = predictions[0]
        print (drived_acc)
        #Save these decoded values with One HotVector into a file.
        # save into a csv file
        gradWriter.writerow({'num_layers' : n_layers, 'num_node1': num_node1, 
                        'num_node2': num_node2, 
                        'F1':f1,'F2':f2,'F3':f3,'F4': f4,'F5':f5,
                        'F6':f6,'F7':f7,'F8':f8,'F9':f9,'F10':f10,
                        'test_acc' : drived_acc})
myGradFile.close()
 
# and repeat until i reach the maximum.(10)
#The final network train on     


y_test_array = y_test.to_numpy()
print ('y_test_array.shape:', y_test_array.shape)  # shape 

print ('y_pred.shape:',nn_predictions.shape)   # shape 


print("y_test_array ========")
print(y_test_array)
print(type(y_test_array) )
print(type(nn_predictions) )

mse_test = mean_sqe_pred(y_test_array, nn_predictions)
K.print_tensor(mse_test, message ='x is: mse_test')  
print (model.metrics_names)
test_loss,  test_decoder_accuracy, test_mean_sqe_pred = model.evaluate(X_test, {'network_with_accuracy': y_test[['test_acc', 'num_layers_N', 'num_node1_N', 'num_node2_N']] })
print (model.evaluate(X_test, {'network_with_accuracy': y_test[['test_acc', 'num_layers_N', 'num_node1_N', 'num_node2_N']] }))
print('####################################################################################')
#print('Test accuracy:', test_decoder_accuracy, test_mean_sqe_pred)
print('####################################################################################')
# save into a csv file
writer.writerow({ 'training_loss' : training_loss[-1], 
                  'val_loss' : val_loss[-1], 
                  'decoder_accuracy' : decoder_accuracy_print[-1],
                  'mean_sqe_pred' : mean_sqe_pred_print[-1],
                  'val_decoder_accuracy' : val_decoder_accuracy[-1],
                  'val_mean_sqe_pred' : val_mean_sqe_pred[-1], 
                  'test_loss': test_loss,
                  'test_decoder_accuracy' : test_decoder_accuracy,
                  'test_mean_sqe_pred' : test_mean_sqe_pred
                  })
myFile.close()

print('####################################################################################')
print('# Train for actor-critic #')
print('####################################################################################')

# Now train actor critic
# The main program, Main loop 

with tf.device('GPU:0'):
    #tf.config.experimental.reset_memory_stats('GPU:0')
    file_totalReward = open('/home/bvadhera/huber/rl_results/totalReward.csv', 'w')
    file_avgReward = open('/home/bvadhera/huber/rl_results/avgReward.csv', 'w')
    file_reward = open('/home/bvadhera/huber/rl_results/reward.csv', 'w')
    file_actions = open('/home/bvadhera/huber/rl_results/actions.csv', 'w')
    file_actor_loss = open('/home/bvadhera/huber/rl_results/actor_loss.csv', 'w')   
    file_critic_value = open('/home/bvadhera/huber/rl_results/critic_value.csv', 'w')  
    file_critic_value2 = open('/home/bvadhera/huber/rl_results/critic_value2.csv', 'w') 
    file_critic_loss = open('/home/bvadhera/huber/rl_results/critic_loss.csv', 'w')  
    file_critic_loss2 = open('/home/bvadhera/huber/rl_results/critic_loss2.csv', 'w')
    file_new_policy_actions  = open('/home/bvadhera/huber/rl_results/new_policy_actions.csv', 'w') 
    file_target_actions = open('/home/bvadhera/huber/rl_results/target_actions.csv', 'w')
    file_target_next_state_values = open('/home/bvadhera/huber/rl_results/target_next_state_values.csv', 'w')
    file_target_next_state_values2 = open('/home/bvadhera/huber/rl_results/target_next_state_values2.csv', 'w')
    file_next_state_target_value = open('/home/bvadhera/huber/rl_results/next_state_target_value.csv', 'w')
    file_target_values = open('/home/bvadhera/huber/rl_results/target_values.csv', 'w')
    file_episode_values = open('/home/bvadhera/huber/rl_results/episode_values.csv', 'w')
    file_state_values = open('/home/bvadhera/huber/rl_results/state_values.csv', 'w')
    file_accuracy_values = open('/home/bvadhera/huber/rl_results/accuracy_values.csv', 'w')

    agent = Agent()
    #episods = 100  
    #episods = 2000
    episods = 1000
    ep_reward = []
    total_avgr = []
    target = False
    max_steps_per_episode = 20 # trejectories
    alpha = 0.01
    concat_output = getConcatinateLayerOutput(test_nn, model)
    derived_concat_output =  concat_output[0][0]
    accuracy_state = getAccuracyLayerOutput(concat_output, model)
    for s in range(episods):
        mem_val = tf.config.experimental.get_memory_usage('GPU:0')
        file_episode_values.write(" Episode- " + str(s) + " Start  current Memory usage in bytes: " + str(mem_val)+ " \n")
        #file_episode_values.write(" Episode- " + str(s) + " Start  Peak Memory usage in bytes: " + str(mem_val.get('peak'))+ " \n")
        # Print current time and log into the file for each episode
        print('######################### Print current time episods Starts  ############################################')
        # current date and time in Eastcost
        date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("date and time in East Coast:",date_time+ "\n")	
        print(str(s) + " - episode \n")
        file_episode_values.write(" Episode- " + str(s) + " Start time: " + str(round(time.time() * 1000))+ "\n")	 
        file_episode_values.write(" Episode-ps " + str(s) + " Start time: " + str(round(time.process_time() * 1000))+ "\n")	 
        
        if target == True:
            break
        total_reward = 0 
        state = derived_concat_output
        file_state_values.write(np.array2string(state, precision=8, separator=',') + "\n")
        prev_accuracy = (accuracy_state[0][0][0]).item()
        file_accuracy_values.write(str(prev_accuracy) + " - prev_accuracy\n")
        done = False
        steps_per_episode = 0
        # Training loop is simple i.e it interacts 
        # and stores experiences and learns at each action step.
        while not done: # cretae trejectories 
            # Agent can choose the action based upon the observations of the environment
            file_episode_values.write(" Episode" + str(s) + "step -" + str(steps_per_episode) + " Start time: " + str(round(time.time() * 1000))+ "\n")	 
            file_episode_values.write(" >>>>>>Episode - ps " + str(s) + "step -" + str(steps_per_episode) + " Start time: " + str(round(time.process_time() * 1000))+ "\n")	 
            vars = 0
            action = agent.act(state)
            var_sizes = [np.product(list(map(int, v.get_shape())))*v.dtype.size
                for key in action.graph.get_all_collection_keys() for v in action.graph.get_collection_ref(key)]
            total = sum(var_sizes)/(1024**2)
            #print(total, 'MB')
            file_episode_values.write(">>>> sum of the product of the dimensions of all the variables in action graph: - " + str(total)+ "MB \n")	
            

            #for v in action.all_variables():
                #vars += np.prod(v.get_shape().as_list())
                # vars contains the sum of the product of the dimensions of all the variables in your graph.
                #file_episode_values.write(">>>> sum of the product of the dimensions of all the variables in action graph: - " + str(vars)+ "\n")	
            file_episode_values.write(">>>> Time to get  Action from agent: - " + str(round(time.time() * 1000))+ "\n")	
            file_episode_values.write(">>>>>>>>>> Time-ps to get  Action from agent: - " + str(round(time.process_time() * 1000))+ "\n")	 
            
            actionArray = tf.keras.backend.get_value(action)
            print('######################### actionArray ############################################')
            print(actionArray)
            file_actions.write(np.array2string(actionArray, precision=8, separator=',') + "\n")
            # Get new state, reward and done from the environment
            # Huber how does it take step...where is the execution.
            # How do we simulate in our case step here?
            t1 = round(time.time() * 1000)
            t1_ps = round(time.process_time() * 1000)
            next_state, next_accuracy, reward, done, steps_per_episode  = returnStepValues(action, alpha, state, prev_accuracy, steps_per_episode,max_steps_per_episode, model, done)   
            t2 = round(time.time() * 1000)
            t2_ps = round(time.process_time() * 1000)
            tt_returnStepValues = t2 - t1 
            tt_returnStepValues_ps = t2_ps - t1_ps 
            file_episode_values.write(">>>> Total Time to get returnStepValues in episode: - " + str(s) + "iteration - " + str(steps_per_episode) + " is " + str(tt_returnStepValues)+ "\n")
            file_episode_values.write(">>>>>>>>>>> Total Time in ps to get returnStepValues in episode: - " + str(s) + "iteration - " + str(steps_per_episode) + " is " + str(tt_returnStepValues_ps)+ "\n")

            file_state_values.write(np.array2string(next_state, precision=8, separator=',') + " - next_state\n")
            file_accuracy_values.write(str(next_accuracy) + " - next_accuracy\n")
            print('######################### reward #########################################')
            print(reward)  
            file_reward.write(str(reward)+ "\n")         
            # Function to  next_state = state + alpha * action
            # reward  = diff between the accuracy of (next state and state)
            # done = when is when we acceed 20 steps
            # save this transition
            t3 = round(time.time() * 1000)
            t3_ps = round(time.process_time() * 1000)
            agent.savexp(state, next_state, actionArray, done, reward)
            t4 = round(time.time() * 1000)
            t4_ps = round(time.process_time() * 1000)
            tt_savexp = t4 - t3 
            tt_savexp_ps = t4_ps - t3_ps 
            file_episode_values.write(">>>> Total Time to get savexp in episode: - " + str(s) + "iteration - " + str(steps_per_episode) + " is " + str(tt_savexp)+ "\n")
            file_episode_values.write(">>>>>>>>> Total Time in ps to get savexp in episode: - " + str(s) + "iteration - " + str(steps_per_episode) + " is " + str(tt_savexp_ps)+ "\n")

            # Agent has to learn now
            agent.train(file_target_actions,file_target_next_state_values,file_target_next_state_values2,
                        file_actor_loss,file_critic_value,file_critic_value2, file_critic_loss,file_critic_loss2,
                        file_new_policy_actions,file_next_state_target_value, file_target_values,file_episode_values,s,steps_per_episode)
            t5 = round(time.time() * 1000)
            tt_train = t5 - t4 
            file_episode_values.write(">>>> Total Time to get trained : episode - "  + str(s) + "iteration - " + str(steps_per_episode) + " is " + str(tt_train)+ "\n")
            t5_ps = round(time.time() * 1000)
            tt_train_ps = t5_ps - t4_ps 
            file_episode_values.write(">>>>>>>>>> Total Time in ps to get trained : episode - "  + str(s) + "iteration - " + str(steps_per_episode) + " is " + str(tt_train_ps)+ "\n")
 
            # Save the current state of the environment to the new state
            state = next_state
            prev_accuracy = next_accuracy
            file_state_values.write(np.array2string(state, precision=8, separator=',') + " - state\n")
            file_accuracy_values.write(str(prev_accuracy) + " - prev_accuracy\n")

            #Total score will be added here
            total_reward += reward
            if done:
                file_totalReward.write(str(total_reward) + "\n")
                ep_reward.append(total_reward)
                #calculate the average to get an idea if our agent is learning or not.
                avg_reward = np.mean(ep_reward[-100:])
                avg_rewards_list.append(avg_reward)
                total_avgr.append(avg_reward)
                print("total reward after {} episode is {} and avg reward is {}".format(s, total_reward, avg_reward))
                #date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                t6 = round(time.time() * 1000)
                file_avgReward.write(str(avg_reward)+ " for episode - " +  str(s) + " at " + str(t6) + "\n")
        file_totalReward.flush()
        file_avgReward.flush()
        file_reward.flush()
        file_actions.flush()
        file_actor_loss.flush()   
        file_critic_value.flush() 
        file_critic_value2.flush()
        file_critic_loss.flush()  
        file_critic_loss2.flush()
        file_new_policy_actions.flush()
        file_episode_values.flush()
        file_state_values.flush() 
        file_accuracy_values.flush()  
        file_episode_values.write(" Episode" + str(s) + "iteration - " + str(steps_per_episode)  + "End  time:" + str(round(time.time() * 1000))+ "\n")	 
        
        #file_episode_values.write(" Episode- " + str(s) + " End  current Memory usuage in bytes: " + str(tf.config.experimental.get_memory_usuage('GPU:0'))+ " \n")
         

    file_totalReward.close()
    file_avgReward.close()
    file_reward.close()
    file_actions.close()
    file_actor_loss.close()   
    file_critic_value.close() 
    file_critic_value2.close()
    file_critic_loss.close()  
    file_critic_loss2.close()
    file_new_policy_actions.close()
    file_episode_values.close() 
    file_state_values.close() 
    file_accuracy_values.close() 
## Testing 
'''
total_reward = 0
state = env.reset()
# This is for testing after tarining is done.
while not done:
    action = agent.act(state, True)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    total_reward += reward
    if done:
       print(total_reward)
'''