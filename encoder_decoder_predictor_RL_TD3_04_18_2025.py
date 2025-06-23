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
#from tensorflow.python.keras.backend import set_session
#from tensorflow.python.keras.models import load_model

#tf.compat.v1.disable_eager_execution()
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
 
# Accuracy Matrics First  ERROR ( Need to fix as per huber as we have in between boundry conditions like 1.5 etc)
def decoder_accuracy(y_true,y_pred):
    #predx = K.batch_get_value(y_pred[:,:3])  # tensor for num_layers, node1 & node2 after decoder
    #truex = K.batch_get_value(y_true[:,:3]) # tensor for num_layers, node1 & node2 before decoder
    
    #feature_name*(maxNumOfLayers - minNumOfLayers) + minNumOfLayers
    #Now here minNumOfLayers = 0 
    # maxNumOfLayers - minNumOfLayers = 2
    matrix_multiply = np.array([(multiplyBy,divideBy,divideBy)],dtype = 'float32')
    matrix_multiply = tf.constant(matrix_multiply)
    denorm_y_true = tf.multiply(y_true[:,1:4],  matrix_multiply) 
    denorm_y_pred = tf.multiply(y_pred[:,1:4],  matrix_multiply) 
    
    #denorm_y_true = y_true[:,1:4] * divideBy 
    #denorm_y_pred = y_pred[:,1:4] * divideBy 

    diff = K.equal(K.mean(K.round(denorm_y_true - denorm_y_pred),  axis=-1), 0)
    return(K.cast(diff,tf.float32))

# Accuracy Matrics Second
def mean_sqe_pred(y_true,y_pred): 

    denorm_y_true = y_true[:,:1]
    denorm_y_pred = y_pred[:,:1] 
    return K.square(denorm_y_true - denorm_y_pred)

# Accuracy Matrics Second
def mean_sqe_pred_legal(y_true,y_pred): 

    denorm_y_true = y_true[:,:1]
    y_true_mask = tf.math.divide_no_nan(y_true[:,:1],y_true[:,:1])
    denorm_y_pred = y_pred[:,:1] 
    return K.square(tf.math.multiply((denorm_y_true - denorm_y_pred),y_true_mask) )

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
 
def getStateAndAccuracy(model, divideBy,maxNumOfLayers, minNumOfLayers):
    #Choose randon network to find the best network
    num_layers,num_node1,num_node2 = getRandomLegalNetwork(divideBy,maxNumOfLayers, minNumOfLayers )
    test_nn = pd.DataFrame({'F1': [0], 'F2': [1],'F3': [0], 'F4': [0],'F5': [0],
                'F6': [0],'F7': [0], 'F8': [0], 'F9': [0],'F10': [0], 'num_layers_N': [num_layers],
                'num_node1_N': [num_node1],'num_node2_N': [num_node2]}, index=['501'])
    print("test_nn_fabricated.head() ")
    print(test_nn.head())
    concat_output = getConcatinateLayerOutput(test_nn, model)
    derived_concat_output =  concat_output[0][0]
    accuracy_state = getAccuracyLayerOutput(concat_output, model)
    state = derived_concat_output
    return state, num_layers,num_node1,num_node2, accuracy_state

def decodeEncodedEmbeddings(state):
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
        print (n_layers, num_node1, num_node2)
    return encoded_arr[0][0], encoded_arr[0][1], nLayers, numNode1, numNode2

def getPolicyoutput(index):
    
    total_reward = 0
    max_steps_per_episode = 1000 # trejectories 1000
    alpha = 0.05
   
    #file_accuracy_values_R.write(str(prev_accuracy) + " - prev_accuracy\n")
    done = False
    steps_per_episode = 0
    qValue = 0.0
    # This is for testing after tarining is done.
    # test for network 2,10,10 expecting accuracy 1
    writeTrejectoriesFile = "/home/bvadhera/huber/38234EpisodsRandomRLRandomPolicyRandom100NN/policyTrejectories" + str(index) +".csv"
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
            qActions_1 = (tf.keras.backend.get_value(action))[0][0]
            qActions_2 = (tf.keras.backend.get_value(action))[0][1]
            next_state, next_accuracy, reward, done, steps_per_episode  = returnStepValues(action, alpha, state, prev_accuracy, steps_per_episode,max_steps_per_episode, model, done)   
            # Function to  next_state = state + alpha * action
            # reward  = diff between the accuracy of (next state and state)
            # done = when is when we acceed 20 steps
            
            # Save the current state of the environment to the new state
            state = next_state
            prev_accuracy = next_accuracy
            total_reward += reward
            encoded_x, encoded_y,nlayers,numnode1,numnode2 = decodeEncodedEmbeddings(state)
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
                encoded_x, encoded_y,nlayers,numnode1,numnode2 = decodeEncodedEmbeddings(state) 
                layers = nlayers
                nnode1 = numnode1
                nnode2 = numnode2
                return (encoded_x, encoded_y,layers,nnode1,nnode2, prev_accuracy,reward,total_reward,qValue,qActions_1,qActions_2)
                


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

def getAccuracyLayerOutput(concat_output, model):
    accuracy_func = K.function([model.get_layer('dense_8').input],
                                [model.get_layer('dense_11').output])
    accuracy_output = accuracy_func([concat_output])
    print (accuracy_output)
    return accuracy_output  

def getRandomNetwork(dividing_factor,maxNumOfLayers, minNumOfLayers):
    start_point = 0
    x = random.randint(start_point,3)
    x = (x - minNumOfLayers)/(maxNumOfLayers - minNumOfLayers)
    y = random.randint(start_point,24)
    y = y / dividing_factor
    z = random.randint(start_point,24)
    #if x == 1:
    #    z = 0
    #else:
    z = z / dividing_factor
    return x,y,z

# generate random flot value with step of 0.1
def randrange_float(start, stop, step):
    return random.randint(0, int((stop - start) / step)) * step + start

def validateX(x,illigal_value,list_legal_x,list_illlegal_x):
  x_ = 0.0
  # Save  for equvalent value after roundup/down and illigal/legal 
  if (x >=0.6  and  x <= 1.4) or (x >=1.6  and  x <= 2.4): # Legal equivalent to 1.0
    if (x >=0.6  and  x <= 1.4): # Legal equivalent to 1.0
      x_ = 1.0
    elif (x >=1.6  and  x <= 2.4):  # Legal equivalent to 2.0
      x_ = 2.0
    list_legal_x.append(x)
    illigal_value = False
  else: # illLegal 
    illigal_value = True
    x_ = x
    list_illlegal_x.append(x)
  return illigal_value,x_

def validateY(y,illigal_value,list_legal_y, list_illlegal_y):
  y_ = round(y)
  rounded_Diff =  round( abs(y - y_),2)
   
  # Save as touple for equvalent value after roundup/down and illigal/legal 
  if (y > 23.4): # Illigal
    list_illlegal_y.append(y)
    illigal_value = True
    y_ = y
  elif (y < 0.6): # Illigal as first layer can not have nodes = 0
    list_illlegal_y.append(y)
    illigal_value = True
    y_ = y
  elif (rounded_Diff <= 0.4): # round down, legal
    list_legal_y.append(y)
    illigal_value = False
  else:
    list_illlegal_y.append(y)
    illigal_value = True
    y_ = y
  return illigal_value,y_

def validateZ(z,illigal_value,list_legal_z,list_illlegal_z, x_):
  z_ = round(z)
  rounded_Diff =  round( abs(z - z_),2) 
  # Save as touple for equvalent value after roundup/down and illigal/legal 
  if (z > 23.4): # Illigal
    list_illlegal_z.append(z)
    illigal_value = True
    z_ = z
  elif ((z > 0.6) and  (x_ == 1) ):  # Illigal as second layer can not have nodes if x_ = 1 
    list_illlegal_z.append(z)
    illigal_value = True
    z_ = z
  elif ((z < 0.6) and  (x_ == 2) ):  # Illigal as second layer should have nodes if x_ = 2 
    list_illlegal_z.append(z)
    illigal_value = True
    z_ = z
  elif (rounded_Diff <= 0.4): # round down, legal
    list_legal_z.append(z)
    illigal_value = False
  else:
    list_illlegal_z.append(z)
    illigal_value = True
    z_ = z
  return illigal_value,z_

def getRandomLegalNetwork(dividing_factor,maxNumOfLayers, minNumOfLayers):
    start_point = minNumOfLayers
    illigal_value = True
    list_legal_x = []
    list_illlegal_x = []
    list_legal_y = []
    list_illlegal_y = []
    list_legal_z = []
    list_illlegal_z = []
       
    while illigal_value:
        x = randrange_float(start_point, maxNumOfLayers, 0.1)
        illigal_value,x_ = validateX(x,illigal_value,list_legal_x,list_illlegal_x)
    x = (x - minNumOfLayers)/(maxNumOfLayers - minNumOfLayers)
    illigal_value = True

    while illigal_value:
        y = randrange_float(start_point, 24.0, 0.1)
        illigal_value,y_ = validateY(y,illigal_value,list_legal_y,list_illlegal_y)
    y = y / dividing_factor
    illigal_value = True

    
    while illigal_value:
        if x_ == 1:
            z = 0.0
            illigal_value = False
        else:
            z = randrange_float(start_point, 24.0, 0.1)
            illigal_value,z_ = validateZ(z,illigal_value,list_legal_z,list_illlegal_z,x_)
            z = z / dividing_factor
    return x,y,z 

def testNetworkWithAccuracy(count,divideBy,max_accuracy_before_norm, min_accuracy_before_norm, maxNumOfLayers, minNumOfLayers): 
    listx = []
    listy = []
    listz = []
    listacc = []
    illigal_value = False
    for idx in range(count):
        # 50 linearly spaced numbers
        start_point = 0
        x = randrange_float(start_point, maxNumOfLayers, 0.1)  #(0 - 0.5) and (1.4-1.6) (2.5-3.0 ) accuracy 0 illigal
                                       #(0.6-1.4 mapped to 1 accuracy) (1.6-2.4 mapped to 2 accuracy)
        illigal_value_x,x_ = validateX(x,illigal_value,listx,listx)
        y = randrange_float(start_point, 24.0, 0.1)
        illigal_value_y,y_  = validateY(y,illigal_value,listy,listy)
        z = randrange_float(start_point, 24.0, 0.1)
        illigal_value_z,z_  = validateZ(z,illigal_value,listz,listz,x_)

        # if illigal network then generate Min accuracy
        if ( illigal_value_x or  illigal_value_y or  illigal_value_z ):
            accuracy = -((2)**2 + (12)**2 + (15)**2)
            listacc.append(accuracy) 
        else:
            accuracy = -((x_-2)**2 + (y_-12)**2  + (z_-8)**2 ) 
            listacc.append(accuracy) 


    print (listx)
    print ("/n")
    print (listy)
    print ("/n")
    print (listz)
    print ("/n")
    print (listacc)
    print ("/n")

    listacc = [float(x) for x in listacc] 
    print (listacc)

    dividing_factor = max_accuracy_before_norm - min_accuracy_before_norm
    print (dividing_factor)
    print ("/n")
    listacc[:] = [number - min_accuracy_before_norm for number in listacc]
    print (listacc)
    print ("/n")
    # Normalize to have values between 0 - 1
    listacc_normalized = [x / dividing_factor for x in listacc]

    print (listacc_normalized)

    # Normalize x,y,z
    listx_normalized = [(x - minNumOfLayers)/(maxNumOfLayers - minNumOfLayers) for x in listx]
    listy_normalized = [x / divideBy for x in listy]
    listz_normalized = [x / divideBy for x in listz]
    a_zip = zip(listx_normalized, listy_normalized, listz_normalized, listacc_normalized)
    #a_zip = zip(listx, listy, listz, listacc_normalized)
    zipped_list = list(a_zip)
    
    print(zipped_list)
    return zipped_list


def testLegalNetworkWithAccuracy(count,divideBy,max_accuracy_before_norm, min_accuracy_before_norm, maxNumOfLayers, minNumOfLayers,maxNumOfNodes): 
    list_legal_x = []
    list_illlegal_x = []
    list_legal_y = []
    list_illlegal_y = []
    list_legal_z = []
    list_illlegal_z = []
    list_legal_acc = []
    legal_zipped_list = []
    illigal_value = False

    for idx in range(count):
        # 50 linearly spaced numbers
        start_point = 0
        x = randrange_float(start_point, maxNumOfLayers, 0.1)  #(0 - 0.5) and (1.4-1.6) (2.5-3.0 ) accuracy 0 illigal
                                       #(0.6-1.4 mapped to 1 accuracy) (1.6-2.4 mapped to 2 accuracy)
        illigal_value_x,x_ = validateX(x,illigal_value,list_legal_x,list_illlegal_x)
        y = randrange_float(start_point, 24.0, 0.1)
        illigal_value_y,y_  = validateY(y,illigal_value,list_legal_y,list_illlegal_y)
        z = randrange_float(start_point, 24.0, 0.1)
        illigal_value_z,z_  = validateZ(z,illigal_value,list_legal_z,list_illlegal_z,x_)

        # if illigal network then generate  accuracy
        if ( illigal_value_x or  illigal_value_y or  illigal_value_z ):
            print ("illigal Networks")
        else: #  legal network then generate   accuracy
            accuracy = -((x_-2)**2 + (y_-12)**2  + (z_-8)**2 ) 
            list_legal_acc.append(accuracy)
            toupleLegal = (x,y,z,accuracy)
            legal_zipped_list.append(toupleLegal) 
    zipped_list = []
    
    # Now to get length of both legal and illigal list
    sizeLegalList = len(list_legal_acc) 
    size_legal_zipped_list = len(legal_zipped_list) 
    if sizeLegalList != 0:
        listacc = list_legal_acc
        dividing_factor = max_accuracy_before_norm - min_accuracy_before_norm
        print (dividing_factor)
        print ("/n")
        listacc[:] = [number - min_accuracy_before_norm for number in listacc]
        print (listacc)
        print ("/n")
        # Normalize to have values between 0 - 1
        listacc_normalized = [x / dividing_factor for x in listacc]

        print (listacc_normalized)
        df = pd.DataFrame(legal_zipped_list)
        print(df)
        listx = df.iloc[:,0]
        listy = df.iloc[:,1]
        listz = df.iloc[:,2]
        # Normalize x,y,z
        listx_normalized = [(x - minNumOfLayers)/(maxNumOfLayers - minNumOfLayers) for x in listx]
        listy_normalized = [x / divideBy for x in listy]
        listz_normalized = [x / divideBy for x in listz]
        a_zip = zip(listx_normalized, listy_normalized, listz_normalized, listacc_normalized)
        #a_zip = zip(listx, listy, listz, listacc_normalized)
        zipped_list = list(a_zip)
    
    print(zipped_list)
    return zipped_list

def testAccuracyForListOfNN_Mix(model, count,divideBy, max_accuracy_before_norm, min_accuracy_before_norm, maxNumOfLayers, minNumOfLayers):
    listOfNN_WithAcc = testNetworkWithAccuracy(count,divideBy, max_accuracy_before_norm, min_accuracy_before_norm, maxNumOfLayers, minNumOfLayers)
    listOfNN_WithNN_DrivenAcc = []
    accuracy_error = []
    for i in listOfNN_WithAcc:
        num_layers = i[0]
        num_node1 = i[1]
        num_node2 = i[2]
        acc = i[3] 
        test_nn = pd.DataFrame({'F1': [0], 'F2': [1],'F3': [0], 'F4': [0],'F5': [0],
                'F6': [0],'F7': [0], 'F8': [0], 'F9': [0],'F10': [0], 'num_layers_N': [num_layers],
                'num_node1_N': [num_node1],'num_node2_N': [num_node2]}, index=['10000'])
        print("test_nn_fabricated.head() ")
        print(test_nn.head())
        concat_output = getConcatinateLayerOutput(test_nn, model)
        emb_x =  concat_output[0][0][0]
        emb_y =  concat_output[0][0][1]
        accuracy_state = getAccuracyLayerOutput(concat_output, model)
        ############# try from the entire network
        prediction   = model.predict(test_nn)
        print (prediction[0])
        driven_acc = (accuracy_state[0][0][0]).item()
        print (driven_acc)
        accuracy_by_Formula = acc
        error_accuracy = accuracy_by_Formula - driven_acc
        accuracy_error.append(error_accuracy)
        NN_DrivenAcc = (emb_x,emb_y,num_layers,num_node1,num_node2, driven_acc)
        listOfNN_WithNN_DrivenAcc.append(NN_DrivenAcc)
    # write listOfNN_WithAcc in a csv file
    item_length = len(listOfNN_WithAcc)
    print (item_length)
    with open('/home/bvadhera/huber/38234EpisodsRandomRLRandomPolicyRandom100NN/listOfNN_WithMixAcc.csv', 'w') as test_file:
        file_writer = csv.writer(test_file)
        for i in range(item_length):
            file_writer.writerow([str(element) for element in listOfNN_WithAcc[i]])
    # write listOfNN_WithNN_DrivenAcc in a csv file
    item_length = len(listOfNN_WithNN_DrivenAcc)
    print (item_length)
    with open('/home/bvadhera/huber/38234EpisodsRandomRLRandomPolicyRandom100NN/listOfNN_WithNN_MixDrivenAcc.csv', 'w') as test_file:
        file_writer = csv.writer(test_file)
        for i in range(item_length):
            file_writer.writerow([str(element) for element in listOfNN_WithNN_DrivenAcc[i]])
    # write accuracy_error in a csv file
    item_length = len(accuracy_error)
    print (item_length)
    with open('/home/bvadhera/huber/38234EpisodsRandomRLRandomPolicyRandom100NN/accuracy_error_mix.csv', 'w',newline='') as test_file:
        file_writer = csv.writer(test_file, delimiter='\n')
        file_writer.writerow(accuracy_error)

def testAccuracyForListOfNNLegal(model, count,divideBy, max_accuracy_before_norm, min_accuracy_before_norm, maxNumOfLayers, minNumOfLayers, maxNumOfNodes):
    listOfNN_WithAcc = testLegalNetworkWithAccuracy(count,divideBy, max_accuracy_before_norm, min_accuracy_before_norm, maxNumOfLayers, minNumOfLayers,maxNumOfNodes)
    listOfNN_WithNN_DrivenAcc = []
    accuracy_error = []
    for i in listOfNN_WithAcc:
        num_layers = i[0]
        num_node1 = i[1]
        num_node2 = i[2]
        acc = i[3] 
        test_nn = pd.DataFrame({'F1': [0], 'F2': [1],'F3': [0], 'F4': [0],'F5': [0],
                'F6': [0],'F7': [0], 'F8': [0], 'F9': [0],'F10': [0], 'num_layers_N': [num_layers],
                'num_node1_N': [num_node1],'num_node2_N': [num_node2]}, index=['10010'])
        print("test_nn_fabricated.head() ")
        print(test_nn.head())
        concat_output = getConcatinateLayerOutput(test_nn, model)
        emb_x =  concat_output[0][0][0]
        emb_y =  concat_output[0][0][1]
        accuracy_state = getAccuracyLayerOutput(concat_output, model)
        driven_acc = (accuracy_state[0][0][0]).item()
        accuracy_by_Formula = acc
        error_accuracy = accuracy_by_Formula - driven_acc
        accuracy_error.append(error_accuracy)
        NN_DrivenAcc = (emb_x,emb_y,num_layers,num_node1,num_node2, driven_acc)
        listOfNN_WithNN_DrivenAcc.append(NN_DrivenAcc)
    # write listOfNN_WithAcc in a csv file
    item_length = len(listOfNN_WithAcc)
    print (item_length)
    with open('/home/bvadhera/huber/38234EpisodsRandomRLRandomPolicyRandom100NN/listOfNN_WithLegalAcc.csv', 'w') as test_file:
        file_writer = csv.writer(test_file)
        for i in range(item_length):
            file_writer.writerow([str(element) for element in listOfNN_WithAcc[i]])
    # write listOfNN_WithNN_DrivenAcc in a csv file
    item_length = len(listOfNN_WithNN_DrivenAcc)
    print (item_length)
    with open('/home/bvadhera/huber/38234EpisodsRandomRLRandomPolicyRandom100NN/listOfNN_WithNN_DrivenLegalAcc.csv', 'w') as test_file:
        file_writer = csv.writer(test_file)
        for i in range(item_length):
            file_writer.writerow([str(element) for element in listOfNN_WithNN_DrivenAcc[i]])
    # write accuracy_error in a csv file
    item_length = len(accuracy_error)
    print (item_length)
    with open('/home/bvadhera/huber/38234EpisodsRandomRLRandomPolicyRandom100NN/accuracy_error_legal.csv', 'w',newline='') as test_file:
        file_writer = csv.writer(test_file, delimiter='\n')
        file_writer.writerow(accuracy_error)

def testAccuracyForTrainedListOfNN(model, testDataToCompareAccuracies):
    listOfNN_WithAcc = testDataToCompareAccuracies.values.tolist()
    listOfNN_WithNN_DrivenAcc = []
    accuracy_error = []
    for i in listOfNN_WithAcc:
        num_layers = i[1]
        num_node1 = i[2]
        num_node2 = i[3]
        acc = i[0] 
        test_nn = pd.DataFrame({'F1': [0], 'F2': [1],'F3': [0], 'F4': [0],'F5': [0],
                'F6': [0],'F7': [0], 'F8': [0], 'F9': [0],'F10': [0], 'num_layers_N': [num_layers],
                'num_node1_N': [num_node1],'num_node2_N': [num_node2]}, index=['10001'])
        print("test_nn_fabricated.head() ")
        print(test_nn.head())
        concat_output = getConcatinateLayerOutput(test_nn, model)
        emb_x =  concat_output[0][0][0]
        emb_y =  concat_output[0][0][1]
        accuracy_state = getAccuracyLayerOutput(concat_output, model)
        driven_acc = (accuracy_state[0][0][0]).item()
        accuracy_by_Formula = acc
        error_accuracy = accuracy_by_Formula - driven_acc
        accuracy_error.append(error_accuracy)
        NN_DrivenAcc = (emb_x,emb_y,num_layers,num_node1,num_node2, driven_acc)
        listOfNN_WithNN_DrivenAcc.append(NN_DrivenAcc)
    # write listOfNN_WithAcc in a csv file
    item_length = len(listOfNN_WithAcc)
    print (item_length)
    with open('/home/bvadhera/huber/38234EpisodsRandomRLRandomPolicyRandom100NN/listOfTrainedNN_WithAcc.csv', 'w') as test_file:
        file_writer = csv.writer(test_file)
        for i in range(item_length):
            file_writer.writerow([str(element) for element in listOfNN_WithAcc[i]])
    # write listOfNN_WithNN_DrivenAcc in a csv file
    item_length = len(listOfNN_WithNN_DrivenAcc)
    print (item_length)
    with open('/home/bvadhera/huber/38234EpisodsRandomRLRandomPolicyRandom100NN/listOfTrainedNN_WithNN_DrivenAcc.csv', 'w') as test_file:
        file_writer = csv.writer(test_file)
        for i in range(item_length):
            file_writer.writerow([str(element) for element in listOfNN_WithNN_DrivenAcc[i]])
    # write accuracy_error in a csv file
    item_length = len(accuracy_error)
    print (item_length)
    with open('/home/bvadhera/huber/38234EpisodsRandomRLRandomPolicyRandom100NN/trainedAccuracy_error.csv', 'w',newline='') as test_file:
        file_writer = csv.writer(test_file, delimiter='\n')
        file_writer.writerow(accuracy_error)


def generateEmbeddingsAccuracy(model,grid_DataSet):
    df_grid = pd.read_csv(grid_DataSet)
    print(df_grid.head())
    listOfgridData = df_grid.values.tolist()
    listOfgridData_DrivenAcc = []
    for i in listOfgridData:
        concat_output = i
        emb_x =  concat_output[0]
        emb_y =  concat_output[1]
        ccat = list([np.array([concat_output])])
        accuracy_state = getAccuracyLayerOutput(ccat, model)
        driven_acc = (accuracy_state[0][0][0]).item()
        NN_DrivenAcc = (emb_x,emb_y,driven_acc)
        listOfgridData_DrivenAcc.append(NN_DrivenAcc)
    # write listOfgridData_DrivenAcc in a csv file
    item_length = len(listOfgridData_DrivenAcc)
    print (item_length)
    with open('/home/bvadhera/huber/gridpoints_DrivenAcc_160.csv', 'w') as grid_file:
        file_writer = csv.writer(grid_file)
        for i in range(item_length):
            file_writer.writerow([str(element) for element in listOfgridData_DrivenAcc[i]])

def generateEmbeddingsCritic(grid_DataSet):
    df_grid = pd.read_csv(grid_DataSet)
    print(df_grid.head())
    listOfgridData = df_grid.values.tolist()
    listOfcritic = []
    for i in listOfgridData:
        concat_output = i
        emb_x =  concat_output[0]
        emb_y =  concat_output[1]
        state = np.array(concat_output)
        action = agent.act(state, True)
        # Call Critic
        # convert states and new states, rewards and actions to tensor.
        states = np.array([state])
        actions = np.array([action])
        states = tf.convert_to_tensor(states, dtype= tf.float32)
        actions = tf.convert_to_tensor(actions, dtype= tf.float32)
        qValue = tf.squeeze(agent.critic_main(states, actions), 1)
        qValue = (tf.keras.backend.get_value(qValue))[0]
        NN_DrivenCritic = (emb_x,emb_y,qValue)
        listOfcritic.append(NN_DrivenCritic)
    # write listOfgridData_DrivenAcc in a csv file
    item_length = len(listOfcritic)
    print (item_length)
    with open('/home/bvadhera/huber/gridpoints_DrivenCritic.csv', 'w') as critic_file:
        file_writer = csv.writer(critic_file)
        for i in range(item_length):
            file_writer.writerow([str(element) for element in listOfcritic[i]])
     
def generateEmbeddingsActor(grid_DataSet):
    df_grid = pd.read_csv(grid_DataSet)
    print(df_grid.head())
    listOfgridData = df_grid.values.tolist()
    listOfActor = []
    for i in listOfgridData:
        concat_output = i
        emb_x =  concat_output[0]
        emb_y =  concat_output[1]
        state = np.array(concat_output)
        action = agent.act(state, True)
        qActions_1 = (tf.keras.backend.get_value(action))[0]
        qActions_2 = (tf.keras.backend.get_value(action))[1]
        NN_DrivenActor = (emb_x,emb_y,qActions_1,qActions_2)
        listOfActor.append(NN_DrivenActor)
    # write listOfgridData_DrivenAcc in a csv file
    item_length = len(listOfActor)
    print (item_length)
    with open('/home/bvadhera/huber/gridpoints_DrivenActor.csv', 'w') as actor_file:
        file_writer = csv.writer(actor_file)
        for i in range(item_length):
            file_writer.writerow([str(element) for element in listOfActor[i]])

def testencoderdecoderForTrainedListOfNN():
    listOfNN_WithAcc = testDataToCompareAccuracies.values.tolist()
    encoderDecoderFile = "/home/bvadhera/huber/38234EpisodsRandomRLRandomPolicyRandom100NN/encoderDecoderTest.csv"
    myEncoderDecoderFile = open(encoderDecoderFile, 'w')
    ##  TODO - CHange the headings as per the results
    headerEncoderDecoder = ['elayers','enode1', 'enode2', 'dlayers','dnode1', 'dnode2',] 
    writerEncoderDecoder = csv.DictWriter(myEncoderDecoderFile, fieldnames=headerEncoderDecoder) 
    writerEncoderDecoder.writeheader()
    for i in listOfNN_WithAcc:
        elayers = i[1]
        enode1 = i[2]
        enode2 = i[3] 
        test_enc = pd.DataFrame({'F1': [0], 'F2': [1],'F3': [0], 'F4': [0],'F5': [0],
                'F6': [0],'F7': [0], 'F8': [0], 'F9': [0],'F10': [0], 'num_layers_N': [elayers],
                'num_node1_N': [enode1],'num_node2_N': [enode2]}, index=['10001'])
        print("test_nn_fabricated.head() ")
        print(test_enc.head())
        split_test = Lambda(lambda x: tf.split(x,[10,3],axis=1))(test_enc)
        ## Test Embeddings and Decode those embeddings
        embeddings = encoder_model.predict(split_test[1])
        ## Test decodings and Decode those embeddings

        decodedOutput = decoder_model(embeddings)
        print ("decodedOutput")
        print (decodedOutput)
        # get the x & y points from concat_output
        decodedOutput_array = np.asarray(decodedOutput)
        dlayers = decodedOutput_array.item(0)
        dnode1 = decodedOutput_array.item(1)
        dnode2 = decodedOutput_array.item(2)
        # save into a csv file
        writerEncoderDecoder.writerow({
                'elayers' : elayers, 
                'enode1' : enode1,
                'enode2' : enode2,
                'dlayers' : dlayers, 
                'dnode1' : dnode1,
                'dnode2' : dnode2
                })
    myEncoderDecoderFile.close()
     
def getGradientFromInitialInputVector(accuracy_model, concat_layer):
    with tf.GradientTape() as tape:
        tape.watch(concat_layer)
        preds = accuracy_model(concat_layer)  
        #model_prediction = accuracy_model.output[:, np.argmax(preds[0])]   
        #concat_layer = model.get_layer("concat_layer")   
        #x = tf.Variable(preds)
        #x = tf.convert_to_tensor(preds, np.float32)    
        #y = tf.convert_to_tensor(concat_output, np.float32)        
    grad = tape.gradient(preds, concat_layer)
    gradients = tf.keras.backend.get_value(grad)
    return gradients[0]

def update_xy_in_grad_direction(grads_value,concat_output):
    #grad_array = np.asarray(grads_value)
    grad_x = grads_value.item(0)
    grad_y = grads_value.item(1)
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

    updated_Combined_arr = np.array([list(Combined_arr)])
    list_combined = []
    list_combined.append(updated_Combined_arr)
    accuracy_next_state = getAccuracyLayerOutput(list_combined, model)   
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
        self.h2 = Dense(10, activation='relu',  name='h2')
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
        self.state_h1 = Dense(24, activation='relu', name='state_h1')
        self.state_h2 = Dense(24, activation='relu', name='state_h2')
        self.critic_output = Dense(1, activation='tanh', name='critic_output')

    #  We have call function for Critic for forward propagation operation.
    def call(self, predict_layer, action):
        x = self.state_h1(tf.concat([predict_layer, action], axis=1))
        x = self.state_h2(x)
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
    self.warmup = 7500
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
      state = tf.convert_to_tensor([state], dtype=tf.float32)

      # now we pass state to the actor network to get the actions out.
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
               file_new_policy_actions,file_next_state_target_value, file_target_values):
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
          print(tf.keras.backend.get_value(target_next_state_values)) 
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
      # Ask Huber  to make sure actor gets updated less times critic
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

            actor_loss = -self.critic_main(states, new_policy_actions) 
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
          # Apply the gradients to actor_main trainable variables.
          self.a_opt.apply_gradients(zip(grads3, self.actor_main.trainable_variables))

      #if self.trainstep % self.replace == 0:
      # Perform soft update on our target network. We use default value of tau = 0.005
      # we update our target networks with a tau of 0.005.
      self.update_target()  
      return target_actions,target_next_state_values,target_next_state_values2,critic_value,critic_value2,next_state_target_value,target_values,critic_loss1,critic_loss2,new_policy_actions,actor_loss   


#
#   CODE FOR NETWORK ONE without RL
#   
# Dense Layers nodes for encoder and decoder

latent_dim = 10
latent_dim1 = 15
latent_dim11 = 20
latent_dim2 = 25
divideBy = 0.0
current_dir  = "/home/bvadhera/huber/"
# load The TrainingDataSet File
#inputNN_Architectures_DataSet = current_dir+"combined_nn_architectures_OneHot_orig.csv"
#inputNN_Architectures_DataSet = current_dir+"testDataNetwork_OneHot.csv"
#inputNN_Architectures_DataSet = current_dir+"testDataNetwork500_2_NN_OneHot.csv"
inputNN_Architectures_DataSet = current_dir+"secondNetwork38234.csv"
minmaxAccuracyFile = current_dir+"secondNetwork38234MinMax.csv"
print (inputNN_Architectures_DataSet)

#writeFile = current_dir + "results_" + "combined_nn_architectures_OneHot.csv"
#writeFile = current_dir + "results_" + "testDataNetworkFixedStartingNN_OneHot.csv"
writeFile = current_dir + "results_" + "secondNetwork38234.csv"
myFile = open(writeFile, 'w')

##  TODO - CHange the headings as per the results
header = ['training_loss','val_loss', 
                      'val_loss', 'decoder_accuracy', 'mean_sqe_pred' , 'mean_sqe_pred_legal',
                      'val_decoder_accuracy', 'val_mean_sqe_pred',
                      'test_loss', 'test_decoder_accuracy','test_mean_sqe_pred','test_acc'] 

writer = csv.DictWriter(myFile, fieldnames=header) 
writer.writeheader()
print("writeFile")
print(writeFile)

# read file to read min and max
fileMinMax = open(minmaxAccuracyFile, 'r')
MinMaxLine = fileMinMax.readline()
fileMinMax.close()
#split string by ,
chunks = MinMaxLine.split(',')
max_accuracy_before_norm = float(chunks[0])
min_accuracy_before_norm = float(chunks[1])
maxNumOfLayers = float(chunks[2])
minNumOfLayers = float(chunks[3])
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
divideBy = getMaxNumOfNodesInDataSet(df) - getMinNumOfNodesInDataSet(df)
multiplyBy = maxNumOfLayers - minNumOfLayers
#Normalize the data for num_layers, num_node1, num_node2 only
normalize(df,normalizeProp,divideBy, minNumOfLayers, maxNumOfLayers)
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
testDataToCompareAccuracies = y

# add three columns of 0 to pad 
# Bhanu Add 000 

# add three columns of 0 to pad 
# Bhanu Add 000 
 
#addPaddingofZero(y)


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

# Instantiate an optimizer.
learning_rate = 0.01
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Create a joint NN with split input layer(s))
input_layer = Input(shape=(13,))
print("input_layer.shape")
print(input_layer.shape)
print("type(input_layer)")
print(type(input_layer) )
split = Lambda(lambda x: tf.split(x,[10,3],axis=1))(input_layer)
#split1 = Lambda( lambda x: tf.split(x,[11,2],axis=1))(input_layer)
print ('split[0].shape:', split[0].shape)  # shape one hot vector
print ('split[1].shape:',split[1].shape)   # shape  num_layers_N  num_node1_N  num_node2_N

# to use them as parellel layers

#+++++++++++++++++++++++++++++++++++++
# Building TRACK - 1
#++++++++++++++++++++++++++++++++
# nn_layer goes int encoder  
# Hidden Layers
inputs_encoder_model=Input((3,))
activation=tf.nn.leaky_relu
encoder_layer_1 = Dense(latent_dim1, activation, input_shape=[3])
#print ('encoder_layer_1.shape:', encoder_layer_1.shape)
encoder_layer_11 = Dense(latent_dim11, activation)
encoder_layer_2 = Dense(latent_dim2, activation) 
#output embedding (x,y) Layer
embedding_layer = Dense(int(2), activation) 


#print ('embedding_layer.shape:', embedding_layer.shape)
encoder_model = Model(inputs=inputs_encoder_model, outputs=[embedding_layer(encoder_layer_2(encoder_layer_11(encoder_layer_1(inputs_encoder_model))))])
encoder_model.compile(optimizer=optimizer, loss=None,metrics=None)
#++++++++++++++++++++++++++++++++
# embedding_layer goes int decoder  
#Hidden Layers
inputs_decoder_model=Input((2,))
decoder_layer_1 = Dense(latent_dim1, activation=tf.nn.leaky_relu) 
decoder_layer_11 = Dense(latent_dim11, activation=tf.nn.leaky_relu)
decoder_layer_2 = Dense(latent_dim2, activation=tf.nn.leaky_relu) 
#output decoded (x,y) Layer
decoded_layer = Dense(int(3), activation=tf.nn.sigmoid) ## To DO Activation 0-1
#print ('decoded_layer.shape:', decoded_layer.shape)

decoder_model = Model(inputs=inputs_decoder_model, outputs=[decoded_layer(decoder_layer_2(decoder_layer_11(decoder_layer_1(inputs_decoder_model))))])
decoder_model.compile(optimizer=optimizer, loss=None,metrics=None)
#++++++++++++++++++++++++++++++++
#predict_layer = Concatenate()([split[0],decoded_layer])
#+++++++++++++++++++++++++++++++++++++
# Building  TRACK - 2
#++++++++++++++++++++++++++++++++
#oneHot_layer = Dense(5)(split[0]) # just need  one hot vector
#print ('oneHot_layer.shape:', oneHot_layer.shape)
#Merge oneHot_layer  back with embedding_layer
#predict_layer = Concatenate()([embedding_layer,split[0]])

# Since split[0] is  num_layers_N  num_node1_N  num_node2_N
# we need only num_node1_N  num_node2_N
#splitAgain = Lambda( lambda x: tf.split(x,[1,2],axis=1))(split[1])
#print ('splitAgain[0].shape:', splitAgain[0].shape)  # shape num_layers_N  (1)
#print ('splitAgain[1].shape:',splitAgain[1].shape)   # shape  num_node1_N  num_node2_N (2)
#predict_layer = Concatenate()([splitAgain[1],split[0]])

#predict_layer = Concatenate()([split1[1],split[0]])
#print ('predict_layer.shape:', predict_layer.shape)

inputs_accuracy_model=Input((12,))
#Hidden Layers
hiddenP_layer_1 = Dense(int(15), activation=tf.nn.leaky_relu)
hiddenP_layer_11 = Dense(int(20), activation=tf.nn.leaky_relu)
#print ('hiddenP_layer_1.shape:', hiddenP_layer_1.shape)
hiddenP_layer_2 = Dense(int(25), activation=tf.nn.leaky_relu)
#print ('hiddenP_layer_2.shape:', hiddenP_layer_2.shape)
#Call Model to get predicted Accuracy as output
accuracy_layer = Dense(int(1), activation='sigmoid')
#print ('accuracy_layer.shape:', accuracy_layer.shape)

accuracy_model = Model(inputs=inputs_accuracy_model, 
                            outputs=[accuracy_layer(hiddenP_layer_2(hiddenP_layer_11(hiddenP_layer_1(inputs_accuracy_model))))])
accuracy_model.compile(optimizer=optimizer, loss=None,metrics=None)

#out put of prediction accuracy_layer

#+++++++++++++++++++++++++++++++++++++
# Building Final  JOINT TRACK  
#++++++++++++++++++++++++++++++++
#encoder_model_Out = encoder_model(split[1])
#embedding_f = K.function([encoder_model.input, K.learning_phase()], activation.get_output)
#embeddings = embedding_f((split[1]))
#decoder_model_Out = decoder_model(encoder_model_Out)

#predict_layer = Concatenate()([encoder_model_Out,split[0]])
#input_accuracy_layer = accuracy_model(predict_layer)


# Connecting embedding layer
encoder_OP = encoder_layer_2(encoder_layer_11(encoder_layer_1(split[1])))
concat_layer = Concatenate()([embedding_layer(encoder_OP),split[0]])
# Merge predict_layer (predicted Accuracy) back with decoded_layer
predict_output = Concatenate(name='network_with_accuracy') ([accuracy_layer(hiddenP_layer_2(hiddenP_layer_11(hiddenP_layer_1 (concat_layer) ) ) ),
                                    decoded_layer(decoder_layer_2(decoder_layer_11(decoder_layer_1(embedding_layer(encoder_OP))))) ] )
#print ('predict_output.shape:', predict_output.shape)

#Call Model to get Final Accuracy as output with custom MSE loss function
##Defining the model by specifying the input and output layers
# model = Model(inputs=input_layer, outputs=[predict_output,actor_output, critic_output])

###### Huber Removed here RL  
model = Model(inputs=[input_layer], outputs=[predict_output])
for idx in range(len(model.layers)):
  print(model.get_layer(index = idx).name)
print(model.summary())
# Instantiate an accuracy metric. custom accuracy  type
#accuracyType = tf.keras.metrics.RootMeanSquaredError()

filepath=current_dir +"acc/" #modelFile+"_"+"acc"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max') #monitor='val_tpr',
callbacks_list_NN = [checkpoint]
#First, call compile to configure the optimizer, loss, and metrics to monitor. 
###### Huber Removed here RL  
model.compile(optimizer=optimizer, loss={'network_with_accuracy': custom_loss},
                     metrics={'network_with_accuracy': [decoder_accuracy , mean_sqe_pred, mean_sqe_pred_legal]}) #, run_eagerly=True)

print (type(y_train))

###### Huber Removed here RL 
#history = model.fit(X_train, {'network_with_accuracy': y_train[['test_acc', 'num_layers_N', 'num_node1_N', 'num_node2_N']] }, validation_split=0.1,  
#                                                                    callbacks=callbacks_list_NN, verbose=2, epochs=7000, batch_size=batchSize)
#print (history.history)


# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
#model.save(current_dir+"my_model.h5") 
# Save the weights Manually
#model.save_weights(filepath + "weights")

# Restore the weights
model.load_weights(filepath + "weights")
# Recreate the exact same model, including its weights and the optimizer
#new_model = tf.keras.models.load_model('my_model.h5')

'''
training_loss = history.history["loss"]
###### Huber Removed here RL 
#network_with_accuracy_loss  = history.history["network_with_accuracy_loss"]
val_loss = history.history["val_loss"]
#val_network_with_accuracy_loss = history.history["val_network_with_accuracy_loss"]
decoder_accuracy_print = history.history["decoder_accuracy"]
mean_sqe_pred_print = history.history["mean_sqe_pred"]
mean_sqe_pred_legal_print = history.history["mean_sqe_pred_legal"] 
val_decoder_accuracy = history.history["val_decoder_accuracy"]
val_mean_sqe_pred = history.history["val_mean_sqe_pred"] 
val_mean_sqe_pred_legal = history.history["val_mean_sqe_pred_legal"] 
'''
print (X_test)
#  I need one NN from X_test
# For now i take ist  NN  where F5 = 1
print ((X_test.loc[X_test['F2'] == 1]).iloc[[0]])
test_nn = (X_test.loc[X_test['F2'] == 1]).iloc[[0]]
print("test_nn.head() ")
print(test_nn.head())
print("test_nn type ")
print(type(test_nn)) 
print ('test_nn.shape:', test_nn.shape)  # shape 

#### pick one valid generated network after normalization 
# line 184 in spreedsheet with accuracy 0.8900804289544236

num_layers,num_node1,num_node2 = getRandomLegalNetwork(divideBy,maxNumOfLayers, minNumOfLayers ) # generate valid normalize network
test_nn = pd.DataFrame({'F1': [0], 'F2': [1],'F3': [0], 'F4': [0],'F5': [0],
                    'F6': [0],'F7': [0], 'F8': [0], 'F9': [0],'F10': [0], 'num_layers_N': [num_layers],
                    'num_node1_N': [num_node1],'num_node2_N': [num_node2]}, index=['50112'])
print("test_nn_fabricated.head() ")
print(test_nn.head())
test_nn_rl = test_nn


# Generate predictions for samples
predictions  = model.predict(test_nn)
split_test = Lambda(lambda x: tf.split(x,[10,3],axis=1))(test_nn)
## Test Embeddings and Decode those embeddings
embeddings = encoder_model.predict(split_test[1])
## Test decodings and Decode those embeddings

decodedOutput = decoder_model(embeddings)

#test
#testA = np.array([[61.64005,-10.593268]])
#decodedOutput = decoder_model(testA)


print ("decodedOutput")
print (decodedOutput)
# get the x & y points from concat_output
decodedOutput_array = np.asarray(decodedOutput)
n_layers = decodedOutput_array.item(0)
num_node1 = decodedOutput_array.item(1)
num_node2 = decodedOutput_array.item(2)
if ((n_layers< 0) or (num_node1< 0) or (num_node2 < 0)):
    print ("decod_output is wrong - Investigate")
concat_layer_test = Concatenate()([tf.convert_to_tensor(embeddings),split_test[0]])


print("predictions ========")
print(predictions[0])
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
        #Do 10 iterations in gradient direction
        for index1 in range(num2):
            grads_value = getGradientFromInitialInputVector(accuracy_model,concat_layer_test)
            emb_x_repeat,emb_y_repeat,concat_output = update_xy_in_grad_direction(grads_value,concat_output)
            x =  tf.convert_to_tensor(np.array([[emb_x_repeat, emb_y_repeat]],dtype = np.float32))
            concat_layer_test = Concatenate()([x,split_test[0]])


    #-----------------------
        list_x_y_repeat = [emb_x_repeat, emb_y_repeat]
        print (type(list_x_y_repeat))
    
        x_y_repeat_T = K.constant(np.asarray(list_x_y_repeat), shape=(1,2))
        print (x_y_repeat_T)
        print ('x_y_repeat_T.shape:', x_y_repeat_T.shape) 
        print ('x_y_repeat_T.type:', type(x_y_repeat_T)) 
        #Take enbedding value of that NN and add alpa (0.01) times the gradient 
        #send this through the decoder and get the value of network 
        print ('model.get_layer(dense_3).output.shape:', model.get_layer('dense_3').output.shape) 
        print ('model.get_layer(dense_3).output.type:', type(model.get_layer('dense_3').output) )
        #decod_func = K.function([model.get_layer('dense_2').output],
        #                            [model.get_layer('dense_5').output])
        #decod_output = decod_func([[x_y_repeat_T]])
        decod_output = decoder_model(x_y_repeat_T)
        print ("decod_output")
        print (decod_output)
        print ("TYPE OF decod_output")
        print (type(decod_output))
        # get the x & y points from concat_output
        decod_output_array = np.asarray(decod_output)
        n_layers = decod_output_array.item(0)
        num_node1 = decod_output_array.item(1)
        num_node2 = decod_output_array.item(2)
        if ((n_layers < 0) or (num_node1< 0) or (num_node2 < 0)):
            print ("decod_output is wrong - Investigate")
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
test_loss,  test_decoder_accuracy, test_mean_sqe_pred, test_mean_sqe_pred_legal = model.evaluate(X_test, {'network_with_accuracy': y_test[['test_acc', 'num_layers_N', 'num_node1_N', 'num_node2_N']] })
print (model.evaluate(X_test, {'network_with_accuracy': y_test[['test_acc', 'num_layers_N', 'num_node1_N', 'num_node2_N']] }))
print('####################################################################################')
#print('Test accuracy:', test_decoder_accuracy, test_mean_sqe_pred)
print('####################################################################################')
# save into a csv file
'''
writer.writerow({ 'training_loss' : training_loss[-1], 
                  'val_loss' : val_loss[-1], 
                  'decoder_accuracy' : decoder_accuracy_print[-1],
                  'mean_sqe_pred' : mean_sqe_pred_print[-1],
                  'mean_sqe_pred_legal' : mean_sqe_pred_legal_print[-1],
                  'val_decoder_accuracy' : val_decoder_accuracy[-1],
                  'val_mean_sqe_pred' : val_mean_sqe_pred[-1], 
                  'test_loss': test_loss,
                  'test_decoder_accuracy' : test_decoder_accuracy,
                  'test_mean_sqe_pred' : test_mean_sqe_pred
                  })
'''                 
myFile.close()
print('####################################################################################')
grid_DataSet = current_dir+"gridpoints_160.csv"
#testencoderdecoderForTrainedListOfNN()
#generateEmbeddingsAccuracy(model,grid_DataSet)

print('# Test Accuracy For Hundred NN after training#') # Error generate with new formula and normalizaton

#testAccuracyForListOfNN_Mix(model, 500, divideBy, max_accuracy_before_norm, min_accuracy_before_norm, maxNumOfLayers, minNumOfLayers)

#testAccuracyForListOfNNLegal(model, 500, divideBy, max_accuracy_before_norm, min_accuracy_before_norm, maxNumOfLayers, minNumOfLayers,24 )

print('####################################################################################')

#testAccuracyForTrainedListOfNN(model,testDataToCompareAccuracies)

print('####################################################################################')
print('# Train for actor-critic #')
print('####################################################################################')

agent = Agent()

'''
# Now train actor critic
# The main program, Main loop 
## Phase I use previos test network for every episode.
with tf.device('GPU:0'):
    not_random = False
    if not_random:  # RUN WITH FIXED NETWORK

        # Load weights
        #agent.actor_main.load_weights(filepath + "/non_random/" + "actor_main_weights")
        #agent.actor_target.load_weights(filepath + "/non_random/" + "actor_target_weights")
        #agent.critic_main.load_weights(filepath + "/non_random/" + "critic_main_weights")
        #agent.critic_main2.load_weights(filepath + "/non_random/" + "critic_main2_weights")
        #agent.critic_target.load_weights(filepath + "/non_random/" + "critic_target_weights")
        #agent.critic_target2.load_weights(filepath + "/non_random/" + "critic_target2_weights")

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
        file_state_values = open('/home/bvadhera/huber/rl_results/state_values.csv', 'w')
        file_accuracy_values = open('/home/bvadhera/huber/rl_results/accuracy_values.csv', 'w')
    
       
        #episods = 100  
        #episods = 2000
        episods = 10000
        ep_reward = []
        total_avgr = []
        target = False
        max_steps_per_episode = 300 # trejectories
        alpha = 0.05
        # Take Valid Network
        test_nn = test_nn_rl
        concat_output = getConcatinateLayerOutput(test_nn, model)
        derived_concat_output =  concat_output[0][0]
        accuracy_state = getAccuracyLayerOutput(concat_output, model)
        for s in range(episods):
            mem_val = tf.config.experimental.get_memory_usage('GPU:0')
            print('######################### Print current time episods Starts  ############################################')
            # current date and time in Eastcost
            date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("date and time in East Coast:",date_time+ "\n")	
            print(str(s) + " - episode \n")
    
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
                vars = 0
                action = agent.act(state)

                #action = agent.act(state, evaluate=True) when evaluate policy

                actionArray = tf.keras.backend.get_value(action)
                print('######################### actionArray ############################################')
                print(actionArray)
                file_actions.write(np.array2string(actionArray, precision=8, separator=',') + "\n")
                # Get new state, reward and done from the environment
                # Huber how does it take step...where is the execution.
                # How do we simulate in our case step here?
    
                next_state, next_accuracy, reward, done, steps_per_episode  = returnStepValues(action, alpha, state, prev_accuracy, steps_per_episode,max_steps_per_episode, model, done)   
                file_state_values.write(np.array2string(next_state, precision=8, separator=',') + " - next_state\n")
                file_accuracy_values.write(str(next_accuracy) + " - next_accuracy\n")
                print('######################### reward #########################################')
                print(reward)  
                file_reward.write(str(reward)+ "\n")         
                # Function to  next_state = state + alpha * action
                # reward  = diff between the accuracy of (next state and state)
                # done = when is when we acceed 20 steps
                # save this transition
                agent.savexp(state, next_state, actionArray, done, reward)
                
                # Agent has to learn now
                agent.train(file_target_actions,file_target_next_state_values,file_target_next_state_values2,
                            file_actor_loss,file_critic_value,file_critic_value2, file_critic_loss,file_critic_loss2,
                            file_new_policy_actions,file_next_state_target_value, file_target_values)
                
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
                    avg_reward = total_reward / max_steps_per_episode
                    avg_rewards_list.append(avg_reward)
                    total_avgr.append(avg_reward)
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
            file_state_values.flush() 
            file_accuracy_values.flush() 
            file_target_next_state_values2.flush() 
            file_next_state_target_value.flush()
            file_target_values.flush()

        # Save weights in a file
        agent.actor_main.save_weights(filepath + "/non_random/" + "actor_main_weights")
        agent.actor_target.save_weights(filepath + "/non_random/" + "actor_target_weights")
        agent.critic_main.save_weights(filepath + "/non_random/" + "critic_main_weights")
        agent.critic_main2.save_weights(filepath + "/non_random/" + "critic_main2_weights")
        agent.critic_target.save_weights(filepath + "/non_random/" + "critic_target_weights")
        agent.critic_target2.save_weights(filepath + "/non_random/" + "critic_target2_weights")


        file_totalReward.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_avgReward.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_reward.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_actions.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_actor_loss.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_critic_value.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_critic_value2.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_critic_loss.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_critic_loss2.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_new_policy_actions.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_state_values.write( "Episode - " +  str(s) + " Finsihed"  + "\n") 
        file_accuracy_values.write( "Episode - " +  str(s) + " Finsihed"  + "\n") 
        file_target_next_state_values2.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_next_state_target_value.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_target_values.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
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
        file_state_values.close() 
        file_accuracy_values.close() 
        file_target_next_state_values2.close()
        file_next_state_target_value.close()
        file_target_values.close()

# Now train actor critic
# The main program, Main loop 

## Phase II use random test network for every episode.
    #tf.config.experimental.reset_memory_stats('GPU:0')
    else:

        # load weights in a file
        #agent.actor_main.load_weights(filepath + "/random/" + "actor_main_weights")
        #agent.actor_target.load_weights(filepath + "/random/" + "actor_target_weights")
        #agent.critic_main.load_weights(filepath + "/random/" + "critic_main_weights")
        #agent.critic_main2.load_weights(filepath + "/random/" + "critic_main2_weights")
        #agent.critic_target.load_weights(filepath + "/random/" + "critic_target_weights")
        #agent.critic_target2.load_weights(filepath + "/random/" + "critic_target2_weights")



        file_totalReward_R = open('/home/bvadhera/huber/rl_results/totalReward_R.csv', 'w')
        file_avgReward_R = open('/home/bvadhera/huber/rl_results/avgReward_R.csv', 'w')
        file_Reward_R = open('/home/bvadhera/huber/rl_results/reward_R.csv', 'w')
        file_actions_R = open('/home/bvadhera/huber/rl_results/actions_R.csv', 'w')
        file_actor_loss_R = open('/home/bvadhera/huber/rl_results/actor_loss_R.csv', 'w')   
        file_critic_value_R = open('/home/bvadhera/huber/rl_results/critic_value_R.csv', 'w')  
        file_critic_value2_R = open('/home/bvadhera/huber/rl_results/critic_value2__R.csv', 'w') 
        file_critic_loss_R = open('/home/bvadhera/huber/rl_results/critic_loss_R.csv', 'w')  
        file_critic_loss2_R = open('/home/bvadhera/huber/rl_results/critic_loss2_R.csv', 'w')
        file_new_policy_actions_R  = open('/home/bvadhera/huber/rl_results/new_policy_actions_R.csv', 'w') 
        file_target_actions_R = open('/home/bvadhera/huber/rl_results/target_actions_R.csv', 'w')
        file_target_next_state_values_R = open('/home/bvadhera/huber/rl_results/target_next_state_values_R.csv', 'w')
        file_target_next_state_values2_R = open('/home/bvadhera/huber/rl_results/target_next_state_values2_R.csv', 'w')
        file_next_state_target_value_R = open('/home/bvadhera/huber/rl_results/next_state_target_value_R.csv', 'w')
        file_target_values_R = open('/home/bvadhera/huber/rl_results/target_values_R.csv', 'w')
        file_state_values_R = open('/home/bvadhera/huber/rl_results/state_values_R.csv', 'w')
        file_accuracy_values_R = open('/home/bvadhera/huber/rl_results/accuracy_values_R.csv', 'w')
    
        
        #episods = 100  
        #episods = 2000
        episods = 10000
        ep_reward = []
        total_avgr = []
        target = False
        max_steps_per_episode = 300 # trejectories
        alpha = 0.05
        test_nn = test_nn_rl
        concat_output = getConcatinateLayerOutput(test_nn, model)
        derived_concat_output =  concat_output[0][0]
        accuracy_state = getAccuracyLayerOutput(concat_output, model)
        for s in range(episods):
            mem_val = tf.config.experimental.get_memory_usage('GPU:0')
            print('######################### Print current time episods Starts  ############################################')
            # current date and time in Eastcost
            date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("date and time in East Coast:",date_time+ "\n")	
            print(str(s) + " - episode \n")
    
            if target == True:
                break
            total_reward = 0 
            state = derived_concat_output
            file_state_values_R.write(np.array2string(state, precision=8, separator=',') + "\n")
            prev_accuracy = (accuracy_state[0][0][0]).item()
            file_accuracy_values_R.write(str(prev_accuracy) + " - prev_accuracy\n")
            done = False
            steps_per_episode = 0
            # Training loop is simple i.e it interacts 
            # and stores experiences and learns at each action step.
            while not done: # cretae trejectories 
                # Agent can choose the action based upon the observations of the environment
                vars = 0
                action = agent.act(state)

                #action = agent.act(state, evaluate=True) when evaluate policy

                actionArray = tf.keras.backend.get_value(action)
                print('######################### actionArray ############################################')
                print(actionArray)
                file_actions_R.write(np.array2string(actionArray, precision=8, separator=',') + "\n")
                # Get new state, reward and done from the environment
                # Huber how does it take step...where is the execution.
                # How do we simulate in our case step here?
    
                next_state, next_accuracy, reward, done, steps_per_episode  = returnStepValues(action, alpha, state, prev_accuracy, steps_per_episode,max_steps_per_episode, model, done)   
    
                file_state_values_R.write(np.array2string(next_state, precision=8, separator=',') + " - next_state\n")
                file_accuracy_values_R.write(str(next_accuracy) + " - next_accuracy\n")
                print('######################### reward #########################################')
                print(reward)  
                file_Reward_R.write(str(reward)+ "\n")         
                # Function to  next_state = state + alpha * action
                # reward  = diff between the accuracy of (next state and state)
                # done = when is when we acceed 20 steps
                # save this transition
                agent.savexp(state, next_state, actionArray, done, reward)
                
                # Agent has to learn now
                agent.train(file_target_actions_R,file_target_next_state_values_R,file_target_next_state_values2_R,
                            file_actor_loss_R,file_critic_value_R,file_critic_value2_R, file_critic_loss_R,file_critic_loss2_R,
                            file_new_policy_actions_R,file_next_state_target_value_R, file_target_values_R)
                
                # Save the current state of the environment to the new state
                state = next_state
                prev_accuracy = next_accuracy
                file_state_values_R.write(np.array2string(state, precision=8, separator=',') + " - state\n")
                file_accuracy_values_R.write(str(prev_accuracy) + " - prev_accuracy\n")

                #Total score will be added here
                total_reward += reward
                if done:
                    file_totalReward_R.write(str(total_reward) + "\n")
                    ep_reward.append(total_reward)
                    #calculate the average to get an idea if our agent is learning or not.
                    avg_reward = total_reward / max_steps_per_episode 
                    avg_rewards_list.append(avg_reward)
                    total_avgr.append(avg_reward)
                    t6 = round(time.time() * 1000)
                    file_avgReward_R.write(str(avg_reward)+ " for episode - " +  str(s) + " at " + str(t6) + "\n")
            file_totalReward_R.flush()
            file_avgReward_R.flush()
            file_Reward_R.flush()
            file_actions_R.flush()
            file_actor_loss_R.flush()   
            file_critic_value_R.flush() 
            file_critic_value2_R.flush()
            file_critic_loss_R.flush()  
            file_critic_loss2_R.flush()
            file_new_policy_actions_R.flush()
            file_state_values_R.flush() 
            file_accuracy_values_R.flush() 
            file_target_next_state_values2_R.flush() 
            file_next_state_target_value_R.flush()
            file_target_values_R.flush()
            #Choose randon network for next episode
            num_layers,num_node1,num_node2 = getRandomLegalNetwork(divideBy,maxNumOfLayers, minNumOfLayers ) # generate valid normalize network
            test_nn = pd.DataFrame({'F1': [0], 'F2': [1],'F3': [0], 'F4': [0],'F5': [0],
                    'F6': [0],'F7': [0], 'F8': [0], 'F9': [0],'F10': [0], 'num_layers_N': [num_layers],
                    'num_node1_N': [num_node1],'num_node2_N': [num_node2]}, index=['50112'])
            print("test_nn_fabricated.head() ")
            print(test_nn.head())
            concat_output = getConcatinateLayerOutput(test_nn, model)
            derived_concat_output =  concat_output[0][0]
            accuracy_state = getAccuracyLayerOutput(concat_output, model)

        # Save weights in a file
        agent.actor_main.save_weights(filepath + "/random/" + "actor_main_weights")
        agent.actor_target.save_weights(filepath + "/random/" + "actor_target_weights")
        agent.critic_main.save_weights(filepath + "/random/" + "critic_main_weights")
        agent.critic_main2.save_weights(filepath + "/random/" + "critic_main2_weights")
        agent.critic_target.save_weights(filepath + "/random/" + "critic_target_weights")
        agent.critic_target2.save_weights(filepath + "/random/" + "critic_target2_weights")


        file_totalReward_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_avgReward_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_Reward_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_actions_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_actor_loss_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_critic_value_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_critic_value2_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_critic_loss_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_critic_loss2_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_new_policy_actions_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_state_values_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n") 
        file_accuracy_values_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n") 
        file_target_next_state_values2_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_next_state_target_value_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_target_values_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_totalReward_R.close()
        file_avgReward_R.close()
        file_Reward_R.close()
        file_actions_R.close()
        file_actor_loss_R.close()   
        file_critic_value_R.close() 
        file_critic_value2_R.close()
        file_critic_loss_R.close()  
        file_critic_loss2_R.close()
        file_new_policy_actions_R.close()
        file_state_values_R.close() 
        file_accuracy_values_R.close() 
        file_target_next_state_values2_R.close()
        file_next_state_target_value_R.close()
        file_target_values_R.close()

'''
############################################################################
## Testing the Policy
# This is for testing after tarining is done.
# test for network 2,10,10 expecting accuracy 1
# Which is 0.08, 0.4, 0.4 and accuracy = 1
total_reward = 0
#-----------------
# If you want to directly load the agent weights the uncomment followings
        # Save weights in a file
agent.actor_main.load_weights(filepath + "/random/" + "actor_main_weights")
agent.actor_target.load_weights(filepath + "/random/" + "actor_target_weights")
agent.critic_main.load_weights(filepath + "/random/" + "critic_main_weights")
agent.critic_main2.load_weights(filepath + "/random/" + "critic_main2_weights")
agent.critic_target.load_weights(filepath + "/random/" + "critic_target_weights")
agent.critic_target2.load_weights(filepath + "/random/" + "critic_target2_weights")


#generateEmbeddingsCritic(grid_DataSet)
generateEmbeddingsActor(grid_DataSet)
#-------------------
random100 = True  # Use random NN of 100 NN to do policy evaluation
writePolicyOutPutFile = "/home/bvadhera/huber/38234EpisodsRandomRLRandomPolicyRandom100NN/policyOutPutFor.csv"
myPolicyOutPutFile = open(writePolicyOutPutFile, 'w')
##  TODO - CHange the headings as per the results
headerPolicyOutPutFile = ['encoded_x','encoded_y', 'layers', 'nnode1','nnode2','accuracy','reward','total_reward','critic_value','actor_value_1','actor_value_2'] 
writerPolicyOutPut = csv.DictWriter(myPolicyOutPutFile, fieldnames=headerPolicyOutPutFile) 
writerPolicyOutPut.writeheader()

if random100:
    #=========
    for index1 in range(100):
        sameStartNode = False
        encoded_x, encoded_y,n_layers,num_node1,num_node2, prev_accuracy,reward,total_reward,qValue,qActions_1, qActions_2 = getPolicyoutput(index1)
        # save into a csv file
        writerPolicyOutPut.writerow({ 'encoded_x' : encoded_x, 
                'encoded_y' : encoded_y, 
                'layers' : n_layers,
                'nnode1' : num_node1,
                'nnode2' : num_node2,
                'accuracy' : prev_accuracy, 
                'reward': reward,
                'total_reward': total_reward,
                'critic_value' : qValue,
                'actor_value_1' : qActions_1,
                'actor_value_2' : qActions_2,
                })
        state, num_layers,num_node1,num_node2, accuracy_state = getStateAndAccuracy(model,divideBy,maxNumOfLayers, minNumOfLayers)
else:
    #  same as testNN
    sameStartNode = True
    encoded_x, encoded_y,n_layers,num_node1,num_node2, prev_accuracy,reward,total_reward,qValue,qActions_1, qActions_2 = getPolicyoutput(index1=0)
    # save into a csv file
    writerPolicyOutPut.writerow({ 'encoded_x' : encoded_x, 
            'encoded_y' : encoded_y, 
            'layers' : n_layers,
            'nnode1' : num_node1,
            'nnode2' : num_node2,
            'accuracy' : prev_accuracy, 
            'reward': reward,
            'total_reward': total_reward,
            'critic_value' : qValue,
            'actor_value_1' : qActions_1,
            'actor_value_2' : qActions_2,
            })
myPolicyOutPutFile.close()
