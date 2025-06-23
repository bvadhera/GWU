# helper functions
import math
from pickle import TRUE
from git import IndexEntry
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

import randomNetworkGeneration


def normal_round(n):
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)


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
def actor_accuracy(y_true,y_pred):
    return(K.zeros(1))

# Accuracy Matrics First
def critic_accuracy(y_true,y_pred):
    return(K.zeros(1))

def getMaxNumOfNodesInDataSet(df_max):
    mLayers = df_max['num_layers'].max()
    #valid layers are only 2
    mLayers = 2
    # iterate as many layers and create node columns 
    i = 0
    maxNumOfNodes = []
    for i in range(int(mLayers)):
      node = "num_node" + str(i+1)
      attrib = df_max[node]
      maxNumOfNodes.append(int(attrib.max()))
    print (max(maxNumOfNodes))
    #get max number of nodes
    return max(maxNumOfNodes)

def getMinNumOfNodesInDataSet(df_max):
    mLayers = df_max['num_layers'].max()
    #valid layers are only 2
    mLayers = 2
    # iterate as many layers and create node columns 
    i = 0
    minNumOfNodes = []
    for i in range(int(mLayers)):
      node = "num_node" + str(i+1)
      attrib = df_max[node]
      minNumOfNodes.append(int(attrib.min()))
    print (min(minNumOfNodes))
    #get max number of nodes
    return min(minNumOfNodes)


def addPaddingofZero(df):
    df["actor_X"] = 0.0
    df["actor_Y"] = 0.0
    df["critic_Value"] = 0.0


def normalize(df_,features, divideBy, minNumOfLayers, maxNumOfLayers):
  print(features)
  for feature_name in features:
    if (feature_name == 'num_layers'):  #  num_layers - 1 / 2-1   so that we get 0 or 1 
        df_[feature_name+"_N"] = (df_[feature_name] - minNumOfLayers)/(maxNumOfLayers - minNumOfLayers)
    else: 
        df_[feature_name+"_N"] = df_[feature_name]/divideBy # 

def dNormalize(df_,features, divideBy, minNumOfLayers, maxNumOfLayers):
  print(features)
  for feature_name in features: 
    if (feature_name == 'num_layers'):
        df_[feature_name+"_N"] = (df_[feature_name] + minNumOfLayers)*(maxNumOfLayers - minNumOfLayers) 
    else:
        df_[feature_name+"_N"] = df_[feature_name]*divideBy
 

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
    xy_valueT = tf.convert_to_tensor(xy_value, dtype=tf.float32)
    next_state = tf.add(xy_valueT, step) 
    oneHotT = tf.convert_to_tensor(oneHot, dtype=tf.float32)
    # Put back oneHotVector
    # Get np array from tensor append the onehot and then get the tensor back
    Combined_arr_T = tf.concat([next_state, oneHotT], axis = 0)
    Combined_arr = tf.keras.backend.get_value(Combined_arr_T)

    updated_Combined_arr = np.array([list(Combined_arr)])
    list_combined = []
    list_combined.append(updated_Combined_arr)
    accuracy_next_state = randomNetworkGeneration.getAccuracyLayerOutput(list_combined, model)   
    next_accuracy = (accuracy_next_state[0][0][0]).item()
    
    reward  =  next_accuracy - prev_accuracy 
    if steps_per_episode == max_steps_per_episode:
        done = True
    steps_per_episode =+ steps_per_episode + 1
    return Combined_arr,next_accuracy, reward, done, steps_per_episode

def getPolicyoutput(index,model,agent,divideBy,maxNumOfLayers, minNumOfLayers, decoder_model,current_dir): 
    
    total_reward = 0
    max_steps_per_episode = 1000 # trejectories 1000
    alpha = 0.05
   
    #file_accuracy_values_R.write(str(prev_accuracy) + " - prev_accuracy\n")
    done = False
    steps_per_episode = 0
    qValue = 0.0
    # This is for testing after tarining is done.
    # test for network 2,10,10 expecting accuracy 1
    writeTrejectoriesFile =  current_dir+"38234EpisodsRandomRLRandomPolicyRandom100NN/policyTrejectories" + str(index) +".csv" 
    myTrejectoriesFile = open(writeTrejectoriesFile, 'w')
    ##  TODO - CHange the headings as per the results
    headerTrejectories = ['encoded_x','encoded_y', 'layers', 'nnode1','nnode2','accuracy','reward','total_reward','critic_value','actor_value_1','actor_value_2'] 
    writerTrejectories = csv.DictWriter(myTrejectoriesFile, fieldnames=headerTrejectories) 
    writerTrejectories.writeheader()

    print("writerTrejectories")
    print(writerTrejectories)
    state, num_layers,num_node1,num_node2, accuracy_state = getStateAndAccuracy(model,divideBy,maxNumOfLayers, minNumOfLayers)
    writerTrejectories.writerow({ 'encoded_x' : state[0], 
                'encoded_y' : state[1], 
                'layers' : num_layers,
                'nnode1' : num_node1,
                'nnode2' : num_node2,
                'accuracy' : accuracy_state, 
                'reward': 0.0,
                'total_reward': total_reward,
                'critic_value' : 0.0,
                'actor_value_1'  : 0.0,
                'actor_value_2'  : 0.0
                })
    prev_accuracy = (accuracy_state[0][0][0]).item()

    while not done:
            action = agent.act(state, True)
            # Call Critic
            # ERROR HUBER
            # convert states and new states, rewards and actions to tensor.
            states = np.array([state])
            actions = np.array([action])
            states = tf.convert_to_tensor(states, dtype= tf.float32)
            actions = tf.convert_to_tensor(actions, dtype= tf.float32)
            qValue = tf.squeeze(agent.critic_main(states, actions), 1)
            qValue = (tf.keras.backend.get_value(qValue))[0]
            #get actor value
            #actions = agent.actor_main(states)
            qActions_1 = (tf.keras.backend.get_value(action))[0]
            qActions_2 = (tf.keras.backend.get_value(action))[1]
            next_state, next_accuracy, reward, done, steps_per_episode  = returnStepValues(action, alpha, state, prev_accuracy, steps_per_episode,max_steps_per_episode, model, done)   
            # Function to  next_state = state + alpha * action
            # reward  = diff between the accuracy of (next state and state)
            # done = when is when we acceed 20 steps
            
            # Save the current state of the environment to the new state
            state = next_state
            prev_accuracy = next_accuracy
            total_reward += reward
            encoded_x, encoded_y,nlayers,numnode1,numnode2 = decodeEncodedEmbeddings(state,decoder_model)
            layers = nlayers
            nnode1 = numnode1
            nnode2 = numnode2
            if ((layers < 0) or (nnode1 < 0) or (nnode2 < 0)):
                print ("decod_output is wrong - Investigate")
            # save into a csv file
            writerTrejectories.writerow({ 'encoded_x' : encoded_x, 
                  'encoded_y' : encoded_y, 
                  'layers' : layers,
                  'nnode1' : nnode1,
                  'nnode2' : nnode2,
                  'accuracy' : prev_accuracy, 
                  'reward': reward,
                  'total_reward': total_reward,
                  'critic_value' : qValue,
                  'actor_value_1' : qActions_1,            
                  'actor_value_2'  : qActions_2
                  })
            # decode and save rewards
            if done:
                print(total_reward)
                myTrejectoriesFile.close()
                encoded_x, encoded_y,nlayers,numnode1,numnode2 = decodeEncodedEmbeddings(state, decoder_model) 
                layers = nlayers
                nnode1 = numnode1
                nnode2 = numnode2
                return (encoded_x, encoded_y,layers,nnode1,nnode2, prev_accuracy,reward,total_reward,qValue,qActions_1,qActions_2)
                

def decodeEncodedEmbeddings(state,decoder_model):
    #my_list = np.array([0.00000000309866918541956,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    
    state = list(state) 

    #state = list(my_list)
    
    state = tf.convert_to_tensor([state], dtype=tf.float32)
    split_state_for_encoded_Values = Lambda(lambda x: tf.split(x,[2,10],axis=1))(state)
    encoded_network = split_state_for_encoded_Values[0]
    encoded_arr = tf.keras.backend.get_value(encoded_network)
    #print ("encoded_output")
    #print (encoded_arr[0][0])
    #print (encoded_arr[0][1])
    # send them for decoding
    decod_output = decoder_model(encoded_network)
    # get the x & y points from concat_output
    decod_output_array = np.asarray(decod_output)

    nLayers = decod_output_array.item(0)
    numNode1 = decod_output_array.item(1)
    numNode2 = decod_output_array.item(2)
    print ("decoded_output")
    print (nLayers)
    print (numNode1)
    print (numNode2)
    if ((nLayers< 0) or (numNode1< 0) or (numNode2 < 0)):
        print ("decod_output is wrong - Investigate")
        print (nLayers, numNode1, numNode2)
    return encoded_arr[0][0], encoded_arr[0][1], nLayers, numNode1, numNode2


def getStateAndAccuracy(model, divideBy,maxNumOfLayers, minNumOfLayers):
    #Choose randon network to find the best network
    num_layers,num_node1,num_node2 = randomNetworkGeneration.getRandomLegalNetwork(divideBy,maxNumOfLayers, minNumOfLayers )
    test_nn = pd.DataFrame({'F1': [0], 'F2': [1],'F3': [0], 'F4': [0],'F5': [0],
                'F6': [0],'F7': [0], 'F8': [0], 'F9': [0],'F10': [0], 'num_layers_N': [num_layers],
                'num_node1_N': [num_node1],'num_node2_N': [num_node2]}, index=['501'])
    print("test_nn_fabricated.head() ")
    print(test_nn.head())
    concat_output = randomNetworkGeneration.getConcatinateLayerOutput(test_nn, model)
    derived_concat_output =  concat_output[0][0]
    accuracy_state = randomNetworkGeneration.getAccuracyLayerOutput(concat_output, model)
    state = derived_concat_output
    return state, num_layers,num_node1,num_node2, accuracy_state
