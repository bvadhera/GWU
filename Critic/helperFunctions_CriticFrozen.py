# helper functions
import math
import gc
import statistics
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
from datetime import datetime
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input,Dense,Lambda,Concatenate 
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import uuid
#from pympler import asizeof
#from pympler.tracker import SummaryTracker 

import randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained
#import forwardLookupTree

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

def normalizeTwice(df_,features, divideBy, minNumOfLayers, maxNumOfLayers):
  print(features)
  for feature_name in features:
    if (feature_name == 'num_layers') or (feature_name == 'num1_layers'):  #  num_layers - 1 / 2-1   so that we get 0 or 1 
        df_[feature_name+"_N"] = (df_[feature_name] - minNumOfLayers)/(maxNumOfLayers - minNumOfLayers)
    elif (feature_name == 'num_node1') or (feature_name == 'num1_node1') or (feature_name == 'num_node2') or (feature_name == 'num1_node2'): 
        df_[feature_name+"_N"] = df_[feature_name]/divideBy # 

def dNormalize(df_,features, divideBy, minNumOfLayers, maxNumOfLayers):
  print(features)
  for feature_name in features: 
    if (feature_name == 'num_layers'):
        df_[feature_name+"_N"] = (df_[feature_name] + minNumOfLayers)*(maxNumOfLayers - minNumOfLayers) 
    else:
        df_[feature_name+"_N"] = df_[feature_name]*divideBy
 
def normalizeFeatureVectoreTwice(X,nonAcc_featureProperties,featureVectorMinMax):
    feature_df = pd.read_csv(featureVectorMinMax)
    print(nonAcc_featureProperties)
    for feature_name in nonAcc_featureProperties:
        row = feature_df.loc[feature_df['feature'] == feature_name] 
        min = row["min"].values[0]
        max = row["max"].values[0]
        a = (X[feature_name] - min)/(max - min)
         
        X[feature_name+"_N"] =  pd.Series(a.values, index=X.index)
        X[feature_name] = X[feature_name+"_N"]
        X.drop([feature_name+"_N"], axis=1)

def stdToFeatures(X,nonAcc_featureProperties,acc_featureProperties):
    NOISE_STD_VECTOR = []

    for feature_name in nonAcc_featureProperties:
        std = statistics.stdev(X[feature_name].tolist())
        #X[feature_name] = X[feature_name]*std*0.25
        NOISE_STD_VECTOR.append(std*0.2)
    for feature_name in acc_featureProperties:
        std = statistics.stdev(X[feature_name].tolist())
        #X[feature_name] = X[feature_name]*std*0.1
        NOISE_STD_VECTOR.append(std*0.1)
    return NOISE_STD_VECTOR






# function to calculate next step for RL network
#   next_state = state + alpha * action
#   reward  = diff between the accuracy of (next state and state)
#   done = when is it acceeds 20 steps ([-0.07533027,  0.07512413]
#
# We will get the max node value by creating a tree with depth3 and 5 nodes 
#  state1 = 2+6 = 8
#  state = 2+13 = 15

def returnStepValues(action, alpha, state, current_accuracy,c_legal, steps_per_episode, max_steps_per_episode,accuracy_model, done, agent,file_utility_values_R):
    step_size_x = action[0]*action[2] * alpha
    step_size_y = action[1]*action[2] * alpha
    #step = action * alpha
    arrayChunk = np.split(state,[1,2])

    x_value =  arrayChunk[0]
    next_state_x_value = x_value + step_size_x
    y_value = arrayChunk[1]
    next_state_y_value = y_value + step_size_y

    oneHot = arrayChunk[2]
    xy_value = np.append(next_state_x_value, next_state_y_value)
    xy_valueT = tf.convert_to_tensor(xy_value, dtype=tf.float32)
    next_state = xy_valueT
    oneHotT = tf.convert_to_tensor(oneHot, dtype=tf.float32)
    # Put back oneHotVector
    # Get np array from tensor append the onehot and then get the tensor back
    NextState_T = tf.concat([next_state, oneHotT], axis = 0)
    NextState = tf.keras.backend.get_value(NextState_T)

    updated_Combined_arr = np.array([list(NextState)])
    list_combined = []
    list_combined.append(updated_Combined_arr)  
    accuracy_next_state = accuracy_model.predict(list_combined)
    next_accuracy = (accuracy_next_state[0][0]).item()
    n_legal = (accuracy_next_state[0][1]).item()
    
    #reward  =  next_accuracy - accuracy_prev 
    if steps_per_episode == max_steps_per_episode:
        done = True
    steps_per_episode =+ steps_per_episode + 1

    current_legal = round(c_legal)  # normalizing it to either 0 or 1 (iiligal or legal)
    next_legal = round(n_legal)  # normalizing it to either 0 or 1 (iiligal or legal)

    if ((current_legal == 1.0) and (next_legal == 1.0)):  #Both current and next states are legal
        # check if our next embeddings are not not beyond (-1, 1) boundry as we are using Siameese network only.
        if ((next_state_x_value <= -1.0) or (next_state_x_value >= 1.0) or (next_state_y_value <= -1.0) or (next_state_y_value >= 1.0)):
             mod_reward = -0.1
             NextState = state
             next_accuracy = current_accuracy
             n_legal = current_legal
        else:
             accuracy_gain = next_accuracy - current_accuracy
             mod_reward =  2.5 * accuracy_gain
    elif next_legal == 0.0: # if  next states is illegal
        mod_reward = -0.1
    elif ((current_legal == 0.0) and (next_legal == 1.0)):  # if  current state is illegal and next states is legal
        mod_reward = 0.0
    else:
        print('Error --- We are in undesirable state')
 
    #Huber TODO
    #mod_reward = 100 * accuracy_gain 
    # For Illigal networks punish by substracting reward by 0.1) 
    
    #reward =  mod_reward - (1.0-round(next_legal))*0.1

    reward =   mod_reward  
    depth = 0

    return NextState,  next_accuracy, n_legal, reward, done, steps_per_episode, depth
     

def getMaxUtilityFromForwardLookupTree(depthTree,leavesPerNode,next_accuracy,alpha,reward,nextState,agent,model):
    #get three list to save all tree nodes
    i = 0  # Root Node
    #tree = forwardLookupTree.Tree()
    listOfTuple = []
    
    rootNodeId = str(0)   # Id for the root node
    #tree.create_node(rootNodeId,reward,reward,nextState,nextState,i,rootNodeId)  # root node
    root_tuple = (rootNodeId,reward,reward,nextState,nextState,i,0)  # root node
    listOfTuple.append(root_tuple)
    # now iterate and create a tree of depth=3 with each node has 5 leaves
    list_top_utilities = []
    list_mid_utilities = []
    list_leaf_utilities = []
    #sizeOf_list_top_utilities  = asizeof.asizeof(list_top_utilities)
    #sizeOf_list_mid_utilities =  asizeof.asizeof(list_mid_utilities)
    #sizeOf_list_leaf_utilities = asizeof.asizeof(list_leaf_utilities)
    #sizeOf_listOfTuple = asizeof.asizeof(listOfTuple)
    #print("start sizeOf_list_top_utilities: ", str(sizeOf_list_top_utilities) +  "\n")  # in bytes
    #print("start sizeOf_list_mid_utilities: ", str(sizeOf_list_mid_utilities) +  "\n")  # in bytes
    #print("start sizeOf_list_leaf_utilities: ", str(sizeOf_list_leaf_utilities) +  "\n")  # in bytes
    #print("start sizeOf_listOfTuple: ", str(sizeOf_listOfTuple) +  "\n")  # in bytes
    levelZeroState = nextState
    levelZeroaccuracy = next_accuracy
    for j in range(int(leavesPerNode +1)):
        if len(list_mid_utilities):
                utility,updated_reward,max_state,depth = updateTreeNode(list_mid_utilities,listOfTuple, 
                                                                  levelOneNodeId, agent.gamma)
                list_top_utilities.append((utility,updated_reward,max_state,depth,levelOneNodeId))
                cleanUpList(list_mid_utilities) # clear the list for next cycle
        if (j < leavesPerNode):
            levelOneNodeId = str(i) + str(j) 
            reward,accuracy,nextState = evaluateNode(alpha,levelZeroaccuracy,levelZeroState,agent,model,j)
            # Create a tree node at depth 1
            #tree.create_node(levelOneNodeId, reward,reward,nextState,nextState,1,levelOneNodeId)
            levelOne_tuple = (levelOneNodeId, reward,reward,nextState,nextState,1,0) 
            listOfTuple.append(levelOne_tuple)
            levelOneAccuracy = accuracy # All nodes below should use this accuracy as previous Accuracy
            levelOneState = nextState
            for k in range(int(leavesPerNode+1)):
                if len(list_leaf_utilities):
                    utility,updated_reward,max_state,depth = updateTreeNode(list_leaf_utilities, listOfTuple,
                                                                   levelTwoNodeId, agent.gamma)
                    list_mid_utilities.append((utility,updated_reward,max_state,depth,levelTwoNodeId))
                    cleanUpList(list_leaf_utilities) # clear the list for next cycle
                if (k <  leavesPerNode):   
                    levelTwoNodeId = str(i) + str(j) + str(k) 
                    reward,accuracy,nextState = evaluateNode(alpha,levelOneAccuracy,levelOneState,agent,model,k)
                    #tree.create_node(levelTwoNodeId,reward,reward,nextState,nextState,2,levelTwoNodeId)     
                    levelTwo_tuple = (levelTwoNodeId,reward,reward,nextState,nextState,2,0) 
                    listOfTuple.append(levelTwo_tuple)                   
                    levelTwoAccuracy = accuracy # All nodes below should use this accuracy as previous Accuracy
                    levelTwoState = nextState
                    for l in range(int(leavesPerNode)):
                        levelThreeNodeId = str(i) + str(j) + str(k) + str(l) 
                        reward,accuracy,nextState = evaluateNode(alpha,levelTwoAccuracy,levelTwoState,agent,model,l)
                        utility = calculateUtility(reward,agent,nextState)
                        #tree.create_node(levelThreeNodeId,reward,reward,nextState,nextState,3,levelThreeNodeId,utility)
                        list_leaf_utilities.append((utility,reward,nextState,3,levelThreeNodeId)) # Only saving reward,nextState,depth
    # max utility
    utility,updated_reward,max_state,depth = updateTreeNode(list_top_utilities,listOfTuple, rootNodeId,agent.gamma)
    #sizeOfTree = tree.size_tree("0")
    #print (sizeOfTree)
    #tree.print_tree_nodes_flat("0")
    #tree.show("0")
    cleanUpList(list_top_utilities)
    cleanUpList(listOfTuple)
    #tree.delete_node("0")   # delete root node
    #del tree  # delete the tree and free up memory
    #asizeof.asizeof(tree)
    #sizeOf_list_top_utilities  = asizeof.asizeof(list_top_utilities)
    #sizeOf_list_mid_utilities =  asizeof.asizeof(list_mid_utilities)
    #sizeOf_list_leaf_utilities = asizeof.asizeof(list_leaf_utilities)
    #sizeOf_listOfTuple = asizeof.asizeof(listOfTuple)
    #print("End sizeOf_list_top_utilities: ", str(sizeOf_list_top_utilities) +  "\n")  # in bytes
    #print("End sizeOf_list_mid_utilities: ", str(sizeOf_list_mid_utilities) +  "\n")  # in bytes
    #print("End sizeOf_list_leaf_utilities: ", str(sizeOf_list_leaf_utilities) +  "\n")  # in bytes
    #print("End sizeOf_listOfTuple: ", str(sizeOf_listOfTuple) +  "\n")  # in bytes
    return max_state, updated_reward,depth,utility

def cleanUpList(list):
    list.clear()

def getMyTuple(listOfTuple,nodeId):
    myTuple = ()
    for index  in  listOfTuple:
        if nodeId in index:
            myTuple = index
    return myTuple

def updateTreeNode(list_leaf_utilities, listOfTuple, nodeId, gamma):
    max_utility_tuple = max(list_leaf_utilities, key=lambda x:x[0])
    utility = max_utility_tuple[0]
    node_touple = getMyTuple(listOfTuple,nodeId)
    prev_reward = max_utility_tuple[1]
    max_state = max_utility_tuple[2]
    depth = max_utility_tuple[3]  #This leaf state of max leaf is preserved 
    updated_reward = node_touple[1]+ (gamma * prev_reward)
    utility = node_touple[1] + (gamma * utility)
    #node_touple[6] =  utility 
    #node_touple[2] = updated_reward 
    #node_touple[5] = depth
    #node_touple[4] = max_state # This leaf state of max leaf is preserved 
    return utility,updated_reward,max_state,depth 

def calculateUtility(reward,agent,nextState):
    action = agent.act(nextState)
    gamma = agent.gamma
    states = np.array([nextState])
    actions = np.array([action])
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.float32)
    critic_value = tf.squeeze( agent.critic_main(states, actions), 1)
    critic_value_state = (tf.keras.backend.get_value(critic_value))[0][0]  ## Raghav  
    critic_value_state_action = (tf.keras.backend.get_value(critic_value))[1][0]
    #Normalizing as we boosted it by 50 factor 
    critic_value_state_action = critic_value_state_action / 50
    critic_value = critic_value_state + critic_value_state_action
    utility = reward + gamma*critic_value 
    return utility

def evaluateNode(alpha,prev_accuracy,state,agent,model,node_num):
    # Dont add npise to the very first node
    if node_num == 0:
        noise = 0.0
    else: 
        noise = tf.random.normal(shape=[1,1], mean=0.0, stddev=0.2)
    action = agent.act(state)
    step_size_x = action[0]*action[2] * alpha + noise
    step_size_y = action[1]*action[2] * alpha + noise
    #step = action * alpha
    
    arrayChunk = np.split(state,[1,2])
    x_value =  arrayChunk[0]
    next_state_x_value = x_value + step_size_x
    y_value = arrayChunk[1]
    next_state_y_value = y_value + step_size_y

    oneHot = arrayChunk[2]
    xy_value = np.append(next_state_x_value, next_state_y_value)
    xy_valueT = tf.convert_to_tensor(xy_value, dtype=tf.float32)
    next_state = xy_valueT
    oneHotT = tf.convert_to_tensor(oneHot, dtype=tf.float32)
    # Put back oneHotVector
    # Get np array from tensor append the onehot and then get the tensor back
    NextState_T = tf.concat([next_state, oneHotT], axis = 0)
    NextState = tf.keras.backend.get_value(NextState_T)

    updated_Combined_arr = np.array([list(NextState)])
    list_combined = []
    list_combined.append(updated_Combined_arr)
    accuracy_next_state = randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.getAccuracyLayerOutput(list_combined, model)   
    next_accuracy = (accuracy_next_state[0][0][0]).item()
    
    reward  =  next_accuracy - prev_accuracy 
    #critic_value = tf.squeeze( agent.critic_main(NextState, action), 1)
    #critic_value_state = (tf.keras.backend.get_value(critic_value))[0][0]
    
    #utility = reward + agent.gamma*critic_value_state
    return  reward,next_accuracy,NextState

def getPolicyoutput(index,model,accuracy_model,agent,divideBy,maxNumOfLayers, minNumOfLayers, decoder_model,current_dir,file_utility_values_R): 
    
    total_reward = 0
    max_steps_per_episode = 1000 # trejectories 1000
    alpha = 0.05
   
    #file_accuracy_values_R.write(str(prev_accuracy) + " - prev_accuracy\n")
    done = False
    steps_per_episode = 0
    qValue1 = 0.0
    qValue2 = 0.0
    # This is for testing after tarining is done.
    # test for network 2,10,10 expecting accuracy 1
    writeTrejectoriesFile =  current_dir+"38234EpisodsRandomRLRandomPolicyRandom100NN/policyTrejectories" + str(index) +".csv" 
    myTrejectoriesFile = open(writeTrejectoriesFile, 'w')
    ##  TODO - CHange the headings as per the results
    headerTrejectories = ['encoded_x','encoded_y', 'layers', 'nnode1','nnode2','accuracy','reward','total_reward','critic_value_state','critic_value_state_action','actor_value_1','actor_value_2','actor_value_3'] 
    writerTrejectories = csv.DictWriter(myTrejectoriesFile, fieldnames=headerTrejectories) 
    writerTrejectories.writeheader()

    print("writerTrejectories")
    print(writerTrejectories)   
    state, num_layers,num_node1,num_node2, accuracy,legal = getStateAndAccuracy(model,decoder_model,accuracy_model,divideBy,maxNumOfLayers, minNumOfLayers)
    #prev_accuracy = (accuracy_state[0][0]).item()
    prev_accuracy = accuracy
    #legal = (accuracy_state[0][1]).item()
    legal = round(legal)
    if legal == 1.0:
        writerTrejectories.writerow({ 'encoded_x' : state[0], 
                    'encoded_y' : state[1], 
                    'layers' : num_layers,
                    'nnode1' : num_node1,
                    'nnode2' : num_node2,
                    'accuracy' : accuracy, 
                    'reward': 0.0,
                    'total_reward': total_reward,
                    'critic_value_state' : 0.0,
                    'critic_value_state_action' : 0.0,
                    'actor_value_1'  : 0.0,
                    'actor_value_2'  : 0.0,
                    'actor_value_3'  : 0.0
                    })


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
            qValue1 = (tf.keras.backend.get_value(qValue))[0][0]
            qValue2 = (tf.keras.backend.get_value(qValue))[1][0]
          
            #get actor value
            #actions = agent.actor_main(states)
            qActions_1 = (tf.keras.backend.get_value(action))[0]
            qActions_2 = (tf.keras.backend.get_value(action))[1]
            qActions_3 = (tf.keras.backend.get_value(action))[2]
            # ASK HUBER RAGHAV - Shall we return Tree step or one step 

            next_state, next_accuracy, n_legal,reward, done, steps_per_episode,depth  = returnStepValues(action, alpha, state, prev_accuracy,legal, steps_per_episode,max_steps_per_episode, accuracy_model, done,agent,file_utility_values_R)   
            # Function to  next_state = state + alpha * action
            # reward  = diff between the accuracy of (next state and state)
            # done = when is when we acceed 20 steps
            
            # Save the current state of the environment to the new state
            state = next_state
            prev_accuracy = next_accuracy
            total_reward += reward
            encoded_x, encoded_y,nlayers,numnode1,numnode2 = decodeEncodedEmbeddings(state,decoder_model,model)
            layers = nlayers
            nnode1 = numnode1
            nnode2 = numnode2
            legal = n_legal
            legal = round(legal)
            if ((layers <= 0) or (nnode1 < 0) or (nnode2 < 0)):
                print ("decod_output is wrong - Investigate")
            # save into a csv file
            if legal == 1.0:
                writerTrejectories.writerow({ 'encoded_x' : encoded_x, 
                    'encoded_y' : encoded_y, 
                    'layers' : layers,
                    'nnode1' : nnode1,
                    'nnode2' : nnode2,
                    'accuracy' : prev_accuracy, 
                    'reward': reward,
                    'total_reward': total_reward,
                    'critic_value_state' : qValue1,
                    'critic_value_state_action' : qValue2,
                    'actor_value_1' : qActions_1,            
                    'actor_value_2'  : qActions_2,
                    'actor_value_3'  : qActions_3
                    })
            # decode and save rewards
            if done:
                print(total_reward)
                myTrejectoriesFile.close()
                encoded_x, encoded_y,nlayers,numnode1,numnode2 = decodeEncodedEmbeddings(state, decoder_model,model) 
                layers = nlayers
                nnode1 = numnode1
                nnode2 = numnode2
                return (encoded_x, encoded_y,layers,nnode1,nnode2, prev_accuracy,reward,total_reward,qValue1,qValue1,qActions_1,qActions_2,qActions_3,legal)
                
def getDecoderLayerOutput(model,decoder_model, firstEmbT):
    #print(decoder_model.summary())  
    decod_func = K.function([model.get_layer('dense_5').input],
                                [model.get_layer('dense_9').output])
    #firstEmbT = tf.convert_to_tensor([firstEmbT], dtype=tf.float32)                            
    decod_output = decod_func([firstEmbT])
    print (decod_output)
    return decod_output


def decodeEncodedEmbeddings(state,decoder_model,model):
    #my_list = np.array([0.00000000309866918541956,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    
    state = list(state) 

    #state = list(my_list)
    
    state = tf.convert_to_tensor([state], dtype=tf.float32)
    split_state_for_encoded_Values = Lambda(lambda x: tf.split(x,[2,10],axis=1))(state)
    encoded_network = split_state_for_encoded_Values[0]
    encoded_arr = tf.keras.backend.get_value(encoded_network)
    decod_output = getDecoderLayerOutput(model,decoder_model, encoded_network)
    #encoded_arr = tf.keras.backend.get_value(encoded_network[0][0])
    #print ("encoded_output")
    #print (encoded_arr[0][0])
    #print (encoded_arr[0][1])
    # send them for decoding
    #decod_output = decoder_model(encoded_network)
    # get the x & y points from concat_output
    decod_output_array = np.asarray(decod_output[0][0][0])

    nLayers = decod_output[0][0][0]
    numNode1 = decod_output[0][0][1]
    numNode2 = decod_output[0][0][2]
    print ("decoded_output")
    print (nLayers)
    print (numNode1)
    print (numNode2)
    if ((nLayers< 0) or (numNode1< 0) or (numNode2 < 0)):
        print ("decod_output is wrong - Investigate")
        print (nLayers, numNode1, numNode2)
    return encoded_arr[0][0], encoded_arr[0][1], nLayers, numNode1, numNode2


def getStateAndAccuracy(model,accuracy_model, decoder_model, divideBy,maxNumOfLayers, minNumOfLayers):

    #Choose randon network to find the best network 
    
    num_layers,num_node1,num_node2 = randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.getRandomLegalNetworkTest(divideBy,maxNumOfLayers, minNumOfLayers )
    
    test_nn = pd.DataFrame({'F1': [0], 'F2': [1],'F3': [0], 'F4': [0],'F5': [1],
                'F6': [0],'F7': [0], 'F8': [0], 'F9': [0],'F10': [0], 'num_layers_N': [num_layers],
                'num_node1_N': [num_node1],'num_node2_N': [num_node2]}, index=['501'])
    '''
    num_layers = 0.666666
    num_node1 = 0.333333
    num_node2 = 0.166666
    test_nn = pd.DataFrame({'F1': [0], 'F2': [1],'F3': [0], 'F4': [0],'F5': [1],
                'F6': [0],'F7': [0], 'F8': [0], 'F9': [0],'F10': [0], 'num_layers_N': [num_layers],
                'num_node1_N': [num_node1],'num_node2_N': [num_node2]}, index=['501'])    
    '''
    print("test_nn_fabricated.head() ")
    print(test_nn.head())
    concat_output = randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.getConcatinateLayerOutput(test_nn, model,decoder_model)
    derived_concat_output =  concat_output[0][0]
    #Hacked
    #concat_output[0][0][1] = -6.869655892
    #concat_output[0][0][0] = -4.99270073
    #Hacked
    model_output = model.predict(test_nn)
    predictions  = model.predict(test_nn)
    accuracy = predictions[0][0]
    legal = predictions[0][1]
    num_layers = predictions[0][2]
    num_node1 = predictions[0][3]
    num_node2 = predictions[0][4]
    emb_x = predictions[0][5]
    emb_y = predictions[0][6]
    #accuracy_state = randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.getAccuracyLayerOutput(concat_output, accuracy_model)
    state = derived_concat_output
    return state, num_layers,num_node1,num_node2, accuracy,legal


