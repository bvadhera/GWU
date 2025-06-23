# random Network generation

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

def testAccuracyForListOfNN_Mix(model, count,divideBy, max_accuracy_before_norm, min_accuracy_before_norm, maxNumOfLayers, minNumOfLayers,current_dir):
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
    with open(current_dir+'38234EpisodsRandomRLRandomPolicyRandom100NN/listOfNN_WithMixAcc.csv', 'w') as test_file:
        file_writer = csv.writer(test_file)
        for i in range(item_length):
            file_writer.writerow([str(element) for element in listOfNN_WithAcc[i]])
    # write listOfNN_WithNN_DrivenAcc in a csv file
    item_length = len(listOfNN_WithNN_DrivenAcc)
    print (item_length)
    with open(current_dir+'38234EpisodsRandomRLRandomPolicyRandom100NN/listOfNN_WithNN_MixDrivenAcc.csv', 'w') as test_file:
        file_writer = csv.writer(test_file)
        for i in range(item_length):
            file_writer.writerow([str(element) for element in listOfNN_WithNN_DrivenAcc[i]])
    # write accuracy_error in a csv file
    item_length = len(accuracy_error)
    print (item_length)
    with open(current_dir+'38234EpisodsRandomRLRandomPolicyRandom100NN/accuracy_error_mix.csv', 'w',newline='') as test_file:
        file_writer = csv.writer(test_file, delimiter='\n')
        file_writer.writerow(accuracy_error)

def testAccuracyForListOfNNLegal(model, count,divideBy, max_accuracy_before_norm, min_accuracy_before_norm, maxNumOfLayers, minNumOfLayers, maxNumOfNodes,current_dir):
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
    with open(current_dir+'38234EpisodsRandomRLRandomPolicyRandom100NN/listOfNN_WithLegalAcc.csv', 'w') as test_file:
        file_writer = csv.writer(test_file)
        for i in range(item_length):
            file_writer.writerow([str(element) for element in listOfNN_WithAcc[i]])
    # write listOfNN_WithNN_DrivenAcc in a csv file
    item_length = len(listOfNN_WithNN_DrivenAcc)
    print (item_length)
    with open(current_dir+'38234EpisodsRandomRLRandomPolicyRandom100NN/listOfNN_WithNN_DrivenLegalAcc.csv', 'w') as test_file:
        file_writer = csv.writer(test_file)
        for i in range(item_length):
            file_writer.writerow([str(element) for element in listOfNN_WithNN_DrivenAcc[i]])
    # write accuracy_error in a csv file
    item_length = len(accuracy_error)
    print (item_length)
    with open(current_dir+'38234EpisodsRandomRLRandomPolicyRandom100NN/accuracy_error_legal.csv', 'w',newline='') as test_file:
        file_writer = csv.writer(test_file, delimiter='\n')
        file_writer.writerow(accuracy_error)

def testAccuracyForTrainedListOfNN(model, testDataToCompareAccuracies,current_dir):
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
    with open(current_dir+'38234EpisodsRandomRLRandomPolicyRandom100NN/listOfTrainedNN_WithAcc.csv', 'w') as test_file:
        file_writer = csv.writer(test_file)
        for i in range(item_length):
            file_writer.writerow([str(element) for element in listOfNN_WithAcc[i]])
    # write listOfNN_WithNN_DrivenAcc in a csv file
    item_length = len(listOfNN_WithNN_DrivenAcc)
    print (item_length)
    with open(current_dir+'38234EpisodsRandomRLRandomPolicyRandom100NN/listOfTrainedNN_WithNN_DrivenAcc.csv', 'w') as test_file:
        file_writer = csv.writer(test_file)
        for i in range(item_length):
            file_writer.writerow([str(element) for element in listOfNN_WithNN_DrivenAcc[i]])
    # write accuracy_error in a csv file
    item_length = len(accuracy_error)
    print (item_length)
    with open(current_dir+'38234EpisodsRandomRLRandomPolicyRandom100NN/trainedAccuracy_error.csv', 'w',newline='') as test_file:
        file_writer = csv.writer(test_file, delimiter='\n')
        file_writer.writerow(accuracy_error)


def generateEmbeddingsAccuracy(model,grid_DataSet,current_dir):
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
    with open(current_dir+'gridpoints_DrivenAcc_160.csv', 'w') as grid_file:
        file_writer = csv.writer(grid_file)
        for i in range(item_length):
            file_writer.writerow([str(element) for element in listOfgridData_DrivenAcc[i]])

def generateEmbeddingsCritic(grid_DataSet, agent,current_dir):
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
    with open(current_dir+'gridpoints_DrivenCritic.csv', 'w') as critic_file:
        file_writer = csv.writer(critic_file)
        for i in range(item_length):
            file_writer.writerow([str(element) for element in listOfcritic[i]])
     
def generateEmbeddingsActor(grid_DataSet,agent,current_dir):
    df_grid = pd.read_csv(grid_DataSet)
    print(df_grid.head())
    listOfgridData = df_grid.values.tolist()
    listOfActor = []
    for i in listOfgridData:
        concat_output = i
        emb_x =  concat_output[0]
        emb_y =  concat_output[1]
        state = np.array(concat_output)
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = agent.actor_main(state)
        #action = agent.act(state, True)
        qActions_1 = (tf.keras.backend.get_value(action))[0][0]
        qActions_2 = (tf.keras.backend.get_value(action))[0][1]
        NN_DrivenActor = (emb_x,emb_y,qActions_1,qActions_2)
        listOfActor.append(NN_DrivenActor)
    # write listOfgridData_DrivenAcc in a csv file
    item_length = len(listOfActor)
    print (item_length)
    with open(current_dir+'gridpoints_DrivenActor.csv', 'w') as actor_file:
        file_writer = csv.writer(actor_file)
        for i in range(item_length):
            file_writer.writerow([str(element) for element in listOfActor[i]])

def generateEmbeddingsActorNew(grid_DataSet,agent,current_dir):
    df_grid = pd.read_csv(grid_DataSet)
    print(df_grid.head())
    listOfgridData = df_grid.values.tolist()
    listOfActor = []
    for i in listOfgridData:
        concat_output = i
        emb_x =  concat_output[0]
        emb_y =  concat_output[1]
        state = np.array(concat_output)
        #state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = agent.act(state, True)
        #action = agent.act(state, True)
        qActions_1 = (tf.keras.backend.get_value(action))[0]
        qActions_2 = (tf.keras.backend.get_value(action))[1]
        NN_DrivenActor = (emb_x,emb_y,qActions_1,qActions_2)
        listOfActor.append(NN_DrivenActor)
    # write listOfgridData_DrivenAcc in a csv file
    item_length = len(listOfActor)
    print (item_length)
    with open(current_dir+'gridpoints_DrivenActor.csv', 'w') as actor_file:
        file_writer = csv.writer(actor_file)
        for i in range(item_length):
            file_writer.writerow([str(element) for element in listOfActor[i]])

     