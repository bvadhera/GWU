import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import gym
from tensorflow.keras.models import load_model


print(tf.config.list_physical_devices('GPU'))

env= gym.make("LunarLanderContinuous-v2")
state_low = env.observation_space.low
state_high = env.observation_space.high
action_low = env.action_space.low 
action_high = env.action_space.high
print(state_low)
print(state_high)
print(action_low)
print(action_high)
avg_rewards_list = []

len(env.action_space.high)

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
    index = self.cnt % self.maxsize
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

# The Network classes

# Critic network class
class Critic(tf.keras.Model):
  def __init__(self):
    super(Critic, self).__init__()
    self.f1 = tf.keras.layers.Dense(512, activation='relu')
    self.f2 = tf.keras.layers.Dense(512, activation='relu')
    self.v =  tf.keras.layers.Dense(1, activation=None)

  #  We have call function for Critic for forward propagation operation.
  #  Since this is for Critic therefore it takes State and action as an input
  def call(self, inputstate, action):
    # We pass the concatinated state & action through our first fully connected layer. 
    # We concatinate along the first axis. The zero'th axis is the batch.  
    x = self.f1(tf.concat([inputstate, action], axis=1))
    x = self.f2(x)
    x = self.v(x)
    return x

# Actor  network class
class Actor(tf.keras.Model):
  def __init__(self, no_action):
    super(Actor, self).__init__()    
    self.f1 = tf.keras.layers.Dense(512, activation='relu')
    self.f2 = tf.keras.layers.Dense(512, activation='relu')
    # We need a actual function which is bound between +1 & -1 
    # as most of our env has an action boundary +/- 1 or its multiple.
    self.mu =  tf.keras.layers.Dense(no_action, activation='tanh')

  #  We have call function for Actor for forward propagation operation.
  def call(self, state):
    x = self.f1(state)
    x = self.f2(x)
    # mu (µ)   If action bounbds are not   +/- 1, can multiply here                                                                      
    x = self.mu(x)  
    return x

 
# Agent network class
# We will need our environment for the max/min actions as 
# we will be adding noise to the output of our deep NN for some exploration.
class Agent():
  # Huber - what is action space here?
  def __init__(self, n_action= len(env.action_space.high)):
    # instantiate out actual networks actor and crtic and target actor and target critic
    # In DDPG, we have target networks for both actor and critic 
    self.actor_main = Actor(n_action)
    self.actor_target = Actor(n_action)
    self.critic_main = Critic()
    self.critic_target = Critic()
    ## Huber Extra Critic
    # 
    self.critic_main2 = Critic()
    self.critic_target2 = Critic()
    # batch size for our memory sampling
    self.batch_size = 64
    # number of our actions
    self.n_actions = len(env.action_space.high)
    # Adam optimizer for actor  
    #self.a_opt = tf.keras.optimizers.Adam(1e-4)
    ## HUber updated
    self.a_opt = tf.keras.optimizers.Adam(0.001)
    
    # Adam optimizer for  critic
    #self.c_opt = tf.keras.optimizers.Adam(1e-4)
    ## HUber updated
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
    # Huber env.observation_space.shape ?
    self.memory = RBuffer(1_00_000, env.observation_space.shape, len(env.action_space.high))
    self.trainstep = 0
    self.replace = 5
    # We need gamma - a discount factor for update equation
    self.gamma = 0.99

    # max/min actions for our environment 
    # Huber What are min/max (-1/+1) ?
    self.min_action = env.action_space.low[0]
    self.max_action = env.action_space.high[0]
    ## Huber updated
    # Note that we have compiled our target networks as we don’t want
    #  to get an error while copying weights from main networks to target networks.
    self.actor_update_steps = 2   # Third, actor-network is trained after every 2 steps.
    self.warmup = 200
    self.actor_target.compile(optimizer=self.a_opt)
    self.critic_target.compile(optimizer=self.c_opt1)
    self.critic_target2.compile(optimizer=self.c_opt2)
    # default value for our soft update tau
    self.tau = 0.005


  # Choose an Action. It will take current state of environment as
  #  input as well as evaluate=False to train vs test. 
  #  Just test agent without adding the noise to get pure deterministic output.
  def act(self, state, evaluate=False):
    # For action selection, first, we convert our state into a tensor and then pass it
    #  to the actor-network.
    # Therefore we convert the state to tensor & add extra dimension to our 
    # observation(state) the batch dimension that is what deep NN expect as input.
    # They expect the batch dimension. 
      state = tf.convert_to_tensor([state], dtype=tf.float32)
      # now we pass state to the actor network to get the actions out.
      actions = self.actor_main(state)
      # For training, we added noise in action and for testing, we will not add any noise.
      if not evaluate:  # if we are training then we want to get some random normal noise
          actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=0.1)
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
      targets1 = self.actor_target.weights

      # We will iterate over the actor weights and append the weight for actor
      # multiplied by tau and add the weight if the target actor multiplied by (1-tau)
      for i, weight in enumerate(self.actor_main.weights):
          weights1.append(weight * tau + targets1[i]*(1-tau))
      # After we go over the loop(every iteration) we set the weights of the target actor 
      # to the list of weights1
      self.actor_target.set_weights(weights1)

      # target for critic , we do the same for the critic network as explained above
      weights2 = []
      targets2 = self.critic_target.weights
      for i, weight in enumerate(self.critic_main.weights):
          weights2.append(weight * tau + targets2[i]*(1-tau))
      self.critic_target.set_weights(weights2)

      # Huber Update  
      weights3 = []
      targets3 = self.critic_target2.weights
      for i, weight in enumerate(self.critic_main2.weights):
        weights3.append(weight * tau + targets3[i]*(1-tau))
      self.critic_target2.set_weights(weights3)

  # We have the learning function to learn where the bulk of the functionality comes in. 
  # We check if our memory is filled till Batch Size. We dont want to learn for less
  #  then batch size. Then only call train function.
  def train(self):
      if self.memory.cnt < self.batch_size:
        return 

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

          # target_actions is the target actor what are the things 
          # we should do for the new states
          target_actions = self.actor_target(next_states)
          # Huber update
          # Actions from the actor’s target network are regularized by adding noise and then clipping 
          # the action in a range of max and min action.

          # Never add noice more then -0.5/0.5
          target_actions += tf.clip_by_value(tf.random.normal(shape=[*np.shape(target_actions)], 
                                    mean=0.0, stddev=0.2), -0.5, 0.5)
           # Never add noice more then -1/1
          target_actions = self.max_action * (tf.clip_by_value(target_actions, self.min_action, 
                                    self.max_action))
          
          # crtic value for new states is shown below as : target critic evaluation 
          # of the next states and target actions and squeeze along the first dimension.
          target_next_state_values = tf.squeeze(self.critic_target(next_states, target_actions), 1)
          
          # Huber update
          target_next_state_values2 = tf.squeeze(self.critic_target2(next_states, target_actions), 1)
          # Our predicted (critic) values are the output of the main critic network which takes
          #  states and actions from the buffer sample.
          # critic value is the value of the current state's with respect to original state
          # and actions the agent actually took during the course of this episode.
          critic_value = tf.squeeze(self.critic_main(states, actions), 1)
          # Huber update
          critic_value2 = tf.squeeze(self.critic_main2(states, actions), 1)

          # A minimum of two networks is taken into consideration for both next state
          #     values and current state values.
          next_state_target_value = tf.math.minimum(target_next_state_values, 
                                                        target_next_state_values2)
          
          # target for the terminal new state is just the reward for every other state
          # it is reward + discounted value of the resulting state according to the target critic network.
          # Then, we apply the Bellman equation to calculate target values 
          # (target_values = rewards + self.gamma * next_state_target_value * done)
          target_values = rewards + self.gamma * next_state_target_value * dones
          
          # Critic loss is then calculated as MSE of target values and predicted values.
          critic_loss1 = tf.keras.losses.MSE(target_values, critic_value)
          # Huber update
          critic_loss2 = tf.keras.losses.MSE(target_values, critic_value2)
            # for critic loss calculate gradients and apply the gradients 
            # Huber what are trainable_variables for critic here ?
      grads1 = tape1.gradient(critic_loss1, self.critic_main.trainable_variables)
      
      # Huber Apply our gradients on same trainable_variables why ?
      self.c_opt1.apply_gradients(zip(grads1, self.critic_main.trainable_variables))
    
      # Huber Update
      grads2 = tape2.gradient(critic_loss2, self.critic_main2.trainable_variables)
      # Apply the gradients to actor trainable variables.
      self.c_opt2.apply_gradients(zip(grads2, self.critic_main2.trainable_variables))
 
      # Huber Update   do not understand this ? 
      # Actor-network is trained after every 2 steps  why ?
      if self.trainstep % self.actor_update_steps == 0:
                
          with tf.GradientTape() as tape3:
              # These are the actions according to actor based upon its current 
              # set of weights. Not based upon the weights it had at the time 
              # whatever the memory we stored in a agent's memory
              new_policy_actions = self.actor_main(states)
              # It is negative ('-') as we are doing gradient ascent.
              # As in policy gradient methods we dont want to use decent 
              # because that will minimize the total score over time. 
              # Rather we would like to maximize the total score over time.
              # Gradient ascent is just negative of gradient decent.

              # Actor loss is calculated as negative of critic main values with 
              # inputs as the main actor predicted actions.
              actor_loss = -self.critic_main(states, new_policy_actions)
              actor_loss = tf.math.reduce_mean(actor_loss)
                # This is how we get the gradient of the critic loss with respect to Meu (µ) parameter 
          # by taking this actor loss which is proportional to the output of the critic network
          # and is coupled. The gradient is non zero because it has this dependency on the output of our 
          # actor networks. Dependence bacause of non-zero gradient comes from the fact that we are taking actions
          #  with respect to actor network, which is calculated according to theatas (Ɵ super µ)
          #  That can effect from here to the critic network.  That's what allows to take the gradient of the output of the 
          # critic network with respect to the variables of the actor network that's how we get coupling.
          # In the paper they applied the chain rule gradient of critic network with actor network
          grads3 = tape3.gradient(actor_loss, self.actor_main.trainable_variables)
          self.a_opt.apply_gradients(zip(grads3, self.actor_main.trainable_variables))


      #if self.trainstep % self.replace == 0:
      # Perform soft update on our target network. We use default value of tau = 0.005
      # we update our target networks with a tau of 0.005.
      self.update_target()
           
      self.trainstep +=1
 
           
      
# The main program, Main loop 
with tf.device('GPU:0'):
  tf.random.set_seed(336699)
  # In “LunarLanderContinuous-v2” an action is 
  # represented as an array [-1 1], so here length is 2.
  agent = Agent(2)
  #episods = 2000
  episods = 2
  ep_reward = []
  total_avgr = []
  target = False

  for s in range(episods):
    if target == True:
      break
    total_reward = 0 
    # Huber what is env here as it has 8 parameters
    # We will have our own env
    state = env.reset()
    done = False
    # Training loop is simple i.e it interacts 
    # and stores experiences and learns at each action step.
    while not done:
      # Agent can choose the action based upon the observations of the environment
      action = agent.act(state)
      # get new state, reward and done from the environment
      # Huber how does it take step...where is the execution.
      # How do we simulate in our case step here?
      next_state, reward, done, _ = env.step(action)
      # save this transition
      agent.savexp(state, next_state, action, done, reward)
      # Agent has to learn now
      agent.train()
      # Save the current state of the environment to the new state
      state = next_state
      #Total score will be added here
      total_reward += reward
      if done:
          ep_reward.append(total_reward)
          #calculate the average to get an idea if our agent is learning or not.
          avg_reward = np.mean(ep_reward[-100:])
          avg_rewards_list.append(avg_reward)
          total_avgr.append(avg_reward)
          print("total reward after {} steps is {} and avg reward is {}".format(s, total_reward, avg_reward))
          if int(avg_reward) == 200:  #Huber what is this 200
            target = True
    

ep = [i  for i in range(len(avg_rewards_list))]
plt.plot( range(len(avg_rewards_list)),avg_rewards_list,'b')
plt.title("Avg Test Aeward Vs Test Episods")
plt.xlabel("Test Episods")
plt.ylabel("Average Test Reward")
plt.grid(True)
plt.show()
total_reward = 0

total_reward = 0
state = env.reset()
while not done:
    action = agent.act(state, True)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    total_reward += reward
    if done:
       print(total_reward)