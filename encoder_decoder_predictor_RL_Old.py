import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
import csv
from csv import DictReader
import os
import random
import math
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input,Dense,Lambda,Concatenate 
from keras.callbacks import ModelCheckpoint
import keras.backend as K
tf.compat.v1.disable_eager_execution()

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
    print (concat_output)
    return concat_output   

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

 


 # Train for actor-critic
 # https://github.com/keras-team/keras-io/blob/master/examples/rl/actor_critic_cartpole.py
def train_actor_critic(model, test_nn):
    #Train
        eps = np.finfo(np.float32).eps.item()  
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        huber_loss = keras.losses.Huber()
        action_probs_history = []
        action_history = []
        critic_value_history = []
        rewards_history = []
        running_reward = 0
        episode_count = 0
        num_actions = 2
        gamma = 0.99  # Discount factor for past rewards
        max_steps_per_episode = 10
        alpha = 0.01
        concat_output = getConcatinateLayerOutput(test_nn, model)
        derived_concat_output =  concat_output[0][0]
        while True:  # Run until solved
            #state = env.reset()
            # Here state is the current embeddings
            state = derived_concat_output
            # get the hot vector and points seperate x & y  

            episode_reward = 0
            with tf.GradientTape() as tape:
                for timestep in range(1, max_steps_per_episode):  # trejectories
                    # env.render(); Adding this line would show the attempts
                    # of the agent in a pop up window.
                    # converts to 1 dim tensor
                    state = tf.convert_to_tensor(state)
                    # input data is 2 dim so it gives 2 dim tensor
                    state = tf.expand_dims(state, 0)

                    # Predict action probabilities and estimated future rewards
                    # from environment state
                    # This is the mean value of x.y embeddings
                    action_probs = getAction_probs(concat_output, model)
                    critic_value = getCritic_probs(concat_output, model)
                    critic_value_history.append({critic_value[0][0][0]})

                    # Sample action from action Gaussian probability distribution in2 dimensional space
                    action = np.random.choice(num_actions, p=np.squeeze(action_probs[0]))
                    # Action is the mean of the policy distribution
                    action_probs_history.append(tf.math.log(action_probs[0, action]))
                   
                    cov = [[0.1, 0], [0, 0.1]] 
                    # we are generateing a distribution
                    x, y = np.random.multivariate_normal(action_probs, cov, 1).T

                    action_history.append(action_probs,x,y)
                    #This is my actual action i picked x,y 
                    # Apply the sampled action in our environment
                    # output of the actor env is my network space 
                    # Run one timestep of the environment's dynamics.
                '''           
                    state = new state
                    reward = calculate previous and new one accuracies and substract
                    rewards_history.append(reward)
                    episode_reward += reward

                    if done:
                        break

                # Update running reward to check condition for solving
                running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

                # Calculate expected value from rewards
                # - At each timestep what was the total reward received after that timestep
                # - Rewards in the past are discounted by multiplying them with gamma
                # - These are the labels for our critic
                returns = []
                discounted_sum = 0
                for r in rewards_history[::-1]:
                    discounted_sum = r + gamma * discounted_sum
                    returns.insert(0, discounted_sum)

                # Normalize
                returns = np.array(returns)
                returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
                returns = returns.tolist()

                # Calculating loss values to update our network
                history = zip(action_probs_history, critic_value_history, returns)
                actor_losses = []
                critic_losses = []
                for log_prob, value, ret in history:
                    # At this point in history, the critic estimated that we would get a
                    # total reward = `value` in the future. We took an action with log probability
                    # of `log_prob` and ended up recieving a total reward = `ret`.
                    # The actor must be updated so that it predicts an action that leads to
                    # high rewards (compared to critic's estimate) with high probability.
                    diff = ret - value
                    # diff in vector
                    lossvalue = x,y  - action_probs
                    actor_losses.append(lossvalue * diff)  # actor loss should be a scaler  ?

                    # The critic must be updated so that it predicts a better estimate of
                    # the future rewards.
                    critic_losses.append(
                        huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
                    )

                # Backpropagation
                loss_value = sum(actor_losses) + sum(critic_losses)
                ## trainable variables ???
                grads = tape.gradient(loss_value, model.trainable_variables)

                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # Clear the loss and reward history
                action_probs_history.clear()
                critic_value_history.clear()
                rewards_history.clear()

            # Log details
            episode_count += 1
            if episode_count % 10 == 0:
                template = "running reward: {:.2f} at episode {}"
                print(template.format(running_reward, episode_count))

            # STopping criteria if policy has convereged
            if running_reward > 50:  # Condition to consider the task solved
                print("Solved at episode {}!".format(episode_count))
                break
            
            '''
    

    

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


#Hidden Layers  # dense_6 & dense_7
hiddenP_layer_1 = Dense(int(10), activation=tf.nn.leaky_relu)(predict_layer)
print ('hiddenP_layer_1.shape:', hiddenP_layer_1.shape)
hiddenP_layer_2 = Dense(int(5), activation=tf.nn.leaky_relu)(hiddenP_layer_1)
print ('hiddenP_layer_2.shape:', hiddenP_layer_2.shape)
#Call Model to get predicted Accuracy as output
accuracy_layer = Dense(int(1), activation='sigmoid')(predict_layer)
print ('accuracy_layer.shape:', accuracy_layer.shape)


#out put of prediction accuracy_layer


#+++++++++++++++++++++++++++++++++++++
# Building Final  JOINT TRACK  
#++++++++++++++++++++++++++++++++
#Merge predict_layer (predicted Accuracy) back with decoded_layer
predict_output = Concatenate(name='network_with_accuracy')([accuracy_layer,decoded_layer])
print ('predict_outpur_layer.shape:', predict_output.shape)

#+++++++++++++++++++++++++++++++++++++
# Building  TRACK - 3 for RL
#++++++++++++++++++++++++++++++++
#actor layers
h1 = Dense(10, activation='relu')(predict_layer)
actor_output = Dense(2, activation='tanh', name='actor_output')(h1)
#++++++++++++++++++++++++++++++++
#critic layers
state_h1 = Dense(24, activation='relu')(predict_layer)
critic_output = Dense(1, activation='relu', name='critic_output')(state_h1)


#Call Model to get Final Accuracy as output with custom MSE loss function
##Defining the model by specifying the input and output layers
# model = Model(inputs=input_layer, outputs=[predict_output,actor_output, critic_output])
#Error Dr Huber ???
model = Model(inputs=input_layer, outputs=[predict_output,actor_output, critic_output])

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

model.compile(optimizer=optimizer, loss={'network_with_accuracy': custom_loss, 
                    'actor_output': custom_loss_actor, 'critic_output' : custom_loss_critic},
                     metrics={'network_with_accuracy': [decoder_accuracy , mean_sqe_pred],
                               'actor_output': actor_accuracy, 'critic_output' :  critic_accuracy})
# Dr Huber check
#model.compile(optimizer=optimizer, loss={'network_with_accuracy': custom_loss, 'actor_output': 'huber_loss', 'critic_output': 'huber_loss'} , 
#                        metrics={'network_with_accuracy':[decoder_accuracy , mean_sqe_pred],'actor_output': mean_sqe_pred, 'critic_output': mean_sqe_pred} )
#history = model.fit(X_train, y_train, validation_split=0.1,  callbacks=callbacks_list_NN, verbose=2, epochs=1, batch_size=batchSize)
print (type(y_train))

history = model.fit(X_train, {'network_with_accuracy': y_train[['test_acc', 'num_layers_N', 'num_node1_N', 'num_node2_N']], 'actor_output': y_train[['actor_X', 'actor_Y']], 'critic_output': y_train[['critic_Value']] }, validation_split=0.1,  
                                                                    callbacks=callbacks_list_NN, verbose=2, epochs=1, batch_size=batchSize)
print (history.history)
training_loss = history.history["loss"]
network_with_accuracy_loss  = history.history["network_with_accuracy_loss"]
val_loss = history.history["val_loss"]
val_network_with_accuracy_loss = history.history["val_network_with_accuracy_loss"]
decoder_accuracy_print = history.history["network_with_accuracy_decoder_accuracy"]
mean_sqe_pred_print = history.history["network_with_accuracy_mean_sqe_pred"]
val_decoder_accuracy = history.history["val_network_with_accuracy_decoder_accuracy"]
val_mean_sqe_pred = history.history["val_network_with_accuracy_mean_sqe_pred"] 
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
print(predictions)
#print(actor)
#print(critic)
#Take the accurecy value out of test_nn
drived_acc = predictions[0][0]
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
        nn_predictions = predictions[0]
        print("nn_predictions ========")
        print(nn_predictions)
        #Take the accurecy value out of test_nn
        drived_acc = predictions[0][0]
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
test_loss, test_network_with_accuracy_loss, test_actor_output_loss, test_critic_output_loss, test_decoder_accuracy, test_mean_sqe_pred, test_actor_output_actor_accuracy, test_critic_output_critic_accuracy = model.evaluate(X_test, {'network_with_accuracy': y_test[['test_acc', 'num_layers_N', 'num_node1_N', 'num_node2_N']], 'actor_output': y_test[['actor_X', 'actor_Y']], 'critic_output': y_test[['critic_Value']] })
print (model.evaluate(X_test, {'network_with_accuracy': y_test[['test_acc', 'num_layers_N', 'num_node1_N', 'num_node2_N']], 'actor_output': y_test[['actor_X', 'actor_Y']], 'critic_output': y_test[['critic_Value']] }))
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
# Train for actor-critic
print('####################################################################################')

train_actor_critic(model, test_nn)
