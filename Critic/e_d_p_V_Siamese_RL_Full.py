from pickle import TRUE
from git import IndexEntry
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
import pandas as pd
import re
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
import time
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from sklearn.utils import shuffle
import csv
from csv import DictReader
import os
import random
import math
#from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input,Dense,Lambda,Concatenate 
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import helperFunctions_CriticFrozen
import randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained
#import reinforcementTraining_JustAccuracyTrainedLamdaForwardLookupTreeActorCriticTrained 
#import reinforcementTraining_AccuracyTrainedCriticTrainedWithInputData
from keras.layers import LeakyReLU
# Helper Files
import actorCriticAgent_Gradient_Trained_gamma01_tau00005Full
import reinforcementTraining_Gradient_Trained_tau00005CriticFrozen 
from keras.layers import LeakyReLU

#tf.compat.v1.disable_eager_execution()
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)
tf.random.set_seed(545654)


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu,True)

# Critic loss
def critic_loss_A(y_A_true, y_A_pred):
    loss =  K.mean(K.square( y_A_true[:,:1] - y_A_pred[:,:1]))
    return loss

# Critic loss
def critic_loss_V(y_V_true, y_V_pred):
    loss =  K.mean(K.square( y_V_true[:,:1] - y_V_pred[:,:1]))
    return loss

def critic_loss_VA(y_VA_true, y_VA_pred):
    loss =  K.mean(K.square( y_VA_true[:,:2] - y_VA_pred[:,:2]))
    return loss
 




# accuracy legal num_layers num_nodes1 num_nodes2  accuracy1 legal1 num1_layers num1_nodes1 num1_nodes2  distance
# Accuracy Matrics Second
def mean_sqe_pred(y_true,y_pred): 
    denorm_y_true = y_true[:,0:1]  
    denorm_y_pred = y_pred[:,0:1] 
    denorm1_y_true = y_true[:,5:6]  
    denorm1_y_pred = y_pred[:,5:6] 
    return K.square(y_true[:,1:2] * (denorm_y_true - denorm_y_pred)) + K.square(y_true[:,6:7] * (denorm1_y_true - denorm1_y_pred))

 
# Not used
# Accuracy Matrics Second
def mean_sqe_pred_legal(y_true,y_pred): 
    denorm_y_true = y_true[:,0:1]
    y_true_mask = tf.math.divide_no_nan(y_true[:,0:1],y_true[:,0:1])
    denorm_y_pred = y_pred[:,0:1] 
    return K.square(tf.math.multiply((denorm_y_true - denorm_y_pred),y_true_mask) )

# Not used
def custom_loss_accuracy(y_true, y_pred):
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
    legal_loss =  tf.keras.losses.binary_crossentropy((y_true[:,1:2]), (y_pred[:,1:2])) 
    acc_loss =  K.mean(K.square(y_true[:,0:1] - y_pred[:,0:1])) + legal_loss
    return acc_loss

# accuracy legal num_layers num_nodes1 num_nodes2  accuracy1 legal1 num1_layers num1_nodes1 num1_nodes2  distance
def binaryCrossEntrphy(y_true,y_pred): 
    bce1 = tf.keras.metrics.binary_crossentropy(y_true[:,1:2], y_pred[:,1:2]) 
    bce2 = tf.keras.metrics.binary_crossentropy(y_true[:,6:7], y_pred[:,6:7])
    return bce1 + bce2

# How often the > 0.5 legal value when the network was legal and how often it was <= 0.5 illigal 
def legal_Network_Pred(y_true, y_pred):
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
    #legal_matric = legal_matric.result()
    #return legal_matric  
    # check with Dr Huber ? 
    diff1 = K.cast(K.equal((K.round(y_true[:,1:2] - y_pred[:,1:2])), 0), tf.float32) 
    diff2 = K.cast(K.equal((K.round(y_true[:,6:7] - y_pred[:,6:7])), 0), tf.float32) 
    diff = diff1 + diff2
    # Dr Huber ask it should be divided by 2 as it can not have vvalue more then 1
    return(diff)  
 
# Not used
def binaryCrossEntrphyLoss(y_true,y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    term_0 = (1 - y_true) * K.log(1 - y_pred + K.epsilon())  
    term_1 = y_true * K.log(y_pred + K.epsilon())
    return -K.mean(term_0 + term_1, axis=0)

    #return tf.keras.losses.binary_crossentropy(y_true[:,1:2], y_pred[:,1:2])

'''
a[start:stop]  # items start through stop-1
a[start:]      # items start through the rest of the array
a[:stop]       # items from the beginning through stop-1
a[:]           # a copy of the whole array

# accuracy legal num_layers num_nodes1 num_nodes2  accuracy1 legal1 num1_layers num1_nodes1 num1_nodes2  distance
'''
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
    weight = 1.0
    weightL = 1.2
    weightD = 1.08
    weightDist = 0.325

    # accuracy legal num_layers num_nodes1 num_nodes2  accuracy1 legal1 num1_layers num1_nodes1 num1_nodes2  distance
    # if both legal then distance_loss = distance, if one legal and other illigal then  distance_loss = -distance, both illigal the  distance_loss = 0
    # round legal values
    
    distance_loss = (  ( ((1.5*(K.cast(y_true[:,1:2],tf.float32))) -1) * (K.cast(y_true[:,6:7],tf.float32)) )   
                           +   ( (1.5*((K.cast(y_true[:,6:7],tf.float32))) -1) * (K.cast(y_true[:,1:2],tf.float32)) ) ) * (K.cast(y_pred[:,10:11],tf.float32))  # y
   

    matrix_multiply = np.array([(3, 24, 24)],dtype = 'float32')
    matrix_multiply = tf.constant(matrix_multiply)
    denorm_y_true = tf.multiply(y_true[:,2:5],  matrix_multiply) 
    denorm_y_pred = tf.multiply(y_pred[:,2:5],  matrix_multiply) 

    # accuracy legal num_layers num_nodes1 num_nodes2
    matrix_multiply = np.array([(3, 24, 24)],dtype = 'float32')
    matrix_multiply = tf.constant(matrix_multiply)

    denorm_y_true = tf.multiply(y_true[:,2:5],  matrix_multiply)
    denorm_y_pred = tf.multiply(y_pred[:,2:5],  matrix_multiply)

    denorm1_y_true = tf.multiply(y_true[:,7:10],  matrix_multiply) 
    denorm1_y_pred = tf.multiply(y_pred[:,7:10],  matrix_multiply) 

    decoder_loss1 = (K.mean(K.square( y_true[:,1:2] * (denorm_y_true  - denorm_y_pred)), axis=-1)) 
    decoder_loss2 = (K.mean(K.square( y_true[:,6:7] * (denorm1_y_true  - denorm1_y_pred)), axis=-1))
    decoder_loss = decoder_loss1 + decoder_loss2
     #  Ignore illigal networks
    acc_loss =  K.square(y_true[:,0:1] - y_pred[:,0:1])  + K.square(y_true[:,5:6] - y_pred[:,5:6]) 
    legal_loss =   (tf.keras.losses.binary_crossentropy((y_true[:,1:2]), (y_pred[:,1:2])))  +   (tf.keras.losses.binary_crossentropy((y_true[:,6:7]), (y_pred[:,6:7])))
    #legal_loss =  weightL * (binaryCrossEntrphyLoss((y_true[:,1:2]), (y_pred[:,1:2])))
    #legal_loss =  weightL * (binary_crossentropy((y_true[:,1:2]), (y_pred[:,1:2])))
    
    #legal_loss =  weightL * K.binary_crossentropy((y_true[:,1:2]), (y_pred[:,1:2]))
    custom_loss =  weightD*(decoder_loss) + weight*(acc_loss) + weightL*legal_loss + weightDist*distance_loss  #  Ignore illigal networks
    return custom_loss


def distance_loss(y_true, y_pred):

    # accuracy legal num_layers num_nodes1 num_nodes2  accuracy1 legal1 num1_layers num1_nodes1 num1_nodes2  distance
    # if both legal then distance_loss = distance, if one legal and other illigal then  distance_loss = -distance, both illigal the  distance_loss = 0
    # round legal values
    
    distance_loss = (  ( ((1.5*(K.cast(y_true[:,1:2],tf.float32))) -1) * (K.cast(y_true[:,6:7],tf.float32)) )   
                           +   ( (1.5*((K.cast(y_true[:,6:7],tf.float32))) -1) * (K.cast(y_true[:,1:2],tf.float32)) ) ) * (K.cast(y_pred[:,10:11],tf.float32))  # y

    return distance_loss



def acc_loss(y_true, y_pred):
     
    acc_loss =  K.square(y_true[:,0:1] - y_pred[:,0:1])  + K.square(y_true[:,5:6] - y_pred[:,5:6]) 
    return acc_loss


def legal_loss(y_true, y_pred):
 
    legal_loss =   (tf.keras.losses.binary_crossentropy((y_true[:,1:2]), (y_pred[:,1:2])))  +   (tf.keras.losses.binary_crossentropy((y_true[:,6:7]), (y_pred[:,6:7])))
 
    return legal_loss



def decoder_accuracy_legal(y_true,y_pred):
    #predx = K.batch_get_value(y_pred[:,:3])  # tensor for num_layers, node1 & node2 after decoder
    #truex = K.batch_get_value(y_true[:,:3]) # tensor for num_layers, node1 & node2 before decoder
    #feature_name*(maxNumOfLayers - minNumOfLayers) + minNumOfLayers
    #Now here minNumOfLayers = 0 
    # maxNumOfLayers - minNumOfLayers = 2
    #a = tf.keras.backend.get_value(y_true)
    #b = tf.keras.backend.get_value(y_pred)
    matrix_multiply = np.array([(3, 24, 24)],dtype = 'float32')
    matrix_multiply = tf.constant(matrix_multiply)
    
    denorm_y_true = K.round(tf.multiply(y_true[:,2:5],  matrix_multiply)) 
    denorm_y_pred = K.round(tf.multiply(y_pred[:,2:5],  matrix_multiply)) 
    #c = tf.keras.backend.get_value(denorm_y_true)
    #d = tf.keras.backend.get_value(denorm_y_pred)
    e = K.round(denorm_y_true - denorm_y_pred)
    #e1 = tf.keras.backend.get_value(e)
    f = K.mean(e,  axis=-1)
    #f1 = tf.keras.backend.get_value(f)
    g = K.equal(f, 0)
    h = tf.cast((g), tf.float32)
    i = K.mean(y_true[:,1:2],  axis=-1)
    j = i * h
    decoder_accuracy_legal = 2* K.cast(j,tf.float32) 

    denorm_y_true = tf.multiply(y_true[:,7:10],  matrix_multiply) 
    denorm_y_pred = tf.multiply(y_pred[:,7:10],  matrix_multiply) 
    #c = tf.keras.backend.get_value(denorm_y_true)
    #d = tf.keras.backend.get_value(denorm_y_pred)
    e = K.round(denorm_y_true - denorm_y_pred)
    #e1 = tf.keras.backend.get_value(e)
    f = K.mean(e,  axis=-1)
    #f1 = tf.keras.backend.get_value(f)
    g = K.equal(f, 0)
    h = tf.cast((g), tf.float32)
    i = K.mean(y_true[:,6:7],  axis=-1)
    j = i * h
    decoder_accuracy_legal1 = 2* K.cast(j,tf.float32) 

    returnValue =  (decoder_accuracy_legal + decoder_accuracy_legal1) /2 
    #p = tf.keras.backend.get_value(returnValue)
    #tf.keras.backend.get_value(grad)
    #diff = y_true[:,1:2] * tf.cast((K.equal(K.mean(K.round(denorm_y_true - denorm_y_pred),  axis=-1), 0) ), tf.float32)
    #return(2 * (K.cast(diff,tf.float32)))
    return returnValue

# Not used
# Accuracy Matrics First  ERROR ( Need to fix as per huber as we have in between boundry conditions like 1.5 etc)
def decoder_accuracy(y_true,y_pred):
    #predx = K.batch_get_value(y_pred[:,:3])  # tensor for num_layers, node1 & node2 after decoder
    #truex = K.batch_get_value(y_true[:,:3]) # tensor for num_layers, node1 & node2 before decoder
    #feature_name*(maxNumOfLayers - minNumOfLayers) + minNumOfLayers
    #Now here minNumOfLayers = 0 
    # maxNumOfLayers - minNumOfLayers = 2
    matrix_multiply = np.array([(3, 24, 24)],dtype = 'float32')
    matrix_multiply = tf.constant(matrix_multiply)
    denorm_y_true = tf.multiply(y_true[:,2:5],  matrix_multiply) 
    denorm_y_pred = tf.multiply(y_pred[:,2:5],  matrix_multiply) 
    diff = K.equal(K.mean(K.round(denorm_y_true - denorm_y_pred),  axis=-1), 0)  
    return(K.cast(diff,tf.float32))

# accuracy legal num_layers num_nodes1 num_nodes2  accuracy1 legal1 num1_layers num1_nodes1 num1_nodes2  distance
def decoder_loss(y_true, y_pred):
    print("type(y_true)")
    print(type(y_true) )
    print("type(y_pred)")
    print(type(y_pred) )
    # convert y_true to tensorflow.python.keras.engine.keras_tensor.KerasTensor
    print ('##################################')  # shape 
    print ('y_true.shape:', y_true.shape)  # shape 
    print ('y_pred.shape:',y_pred.shape)   # shape 
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    matrix_multiply = np.array([(3, 24, 24)],dtype = 'float32')
    matrix_multiply = tf.constant(matrix_multiply)
    denorm_y_true = tf.multiply(y_true[:,2:5],  matrix_multiply) 
    denorm_y_pred = tf.multiply(y_pred[:,2:5],  matrix_multiply) 
    denorm1_y_true = tf.multiply(y_true[:,7:10],  matrix_multiply) 
    denorm1_y_pred = tf.multiply(y_pred[:,7:10],  matrix_multiply)
    loss1 =  K.mean(K.square( y_true[:,1:2] * (denorm_y_true  - denorm_y_pred)), axis=-1) 
    loss2 =  (K.mean(K.square( y_true[:,6:7] * (denorm1_y_true  - denorm1_y_pred)), axis=-1))
    loss = loss1+loss2
    # Return a function
    return loss 



def getEmbeddingLayerOutput( model, test_nn): 
    print(model.summary())
    split = Lambda(lambda x: tf.split(x,[13,3],axis=1))(test_nn)
    
    emb_func = K.function([model.get_layer('dense_2').input],
                                [model.get_layer('dense_6').output])
    emb_output = emb_func([split[1]])
    print (emb_output)
    return emb_output

def getDecoderLayerOutput(model, firstEmbT):
    print(model.summary())  
    decod_func = K.function([model.get_layer('dense_7').input],
                                [model.get_layer('dense_11').output])
    firstEmbT = tf.convert_to_tensor([firstEmbT], dtype=tf.float32)                            
    decod_output = decod_func([firstEmbT])
    print (decod_output)
    return decod_output

# def getAccuracyLayerOutput(concat_output, accuracy_model):
#     print(accuracy_model.summary())
#     accuracy_func = K.function([accuracy_model.get_layer('input_6').input],
#                                 [accuracy_model.get_layer('accuracy_network').output])
#     accuracy_output = accuracy_func([concat_output])
#     return accuracy_output  

def getAccuracyLayerOutput(concat_output, model):
    print(model.summary())
    accuracy_func = K.function([model.get_layer('dense_12').input],
                                [model.get_layer('accuracy_network').output])
    accuracy_output = accuracy_func([concat_output])
    return accuracy_output  
    '''
    for layer in accuracy_model.layers[:9]:
        accuracy_model_weights = layer.get_weights()
        print(accuracy_model_weights) 
    
    for layer in model.layers[:22]:
        model_weights = layer.get_weights() 
        print(model_weights)
    '''

    accuracy_output = accuracy_model.predict(concat_layer)
    #decod_func = K.function([model.get_layer('dense_10').input],
    #                            [model.get_layer('accuracy_network').output])
    #decod_output = decod_func([concat_layer])
    print (accuracy_output)
    return accuracy_output


def generatePredictionsSniffTest(grid_DataSet, model,encoder_model,decoder_model,accuracy_model, current_dir):
    
    df_grid  = pd.read_csv(grid_DataSet)
    #df3 = pd.read_csv(current_dir+"mixedDataFramesSample.csv")
    #df.to_csv(current_dir+"duplicateDF.csv", sep='\t', encoding='utf-8')
    #print(df.head())
    print(df_grid .head())
    normalizeProp = list(df_grid .columns.values)
    # remove 'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'
    randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.removeOneHotVectorTwice(normalizeProp)
    #Also remove 'test_acc' so that we dont normalize the accuracy
    normalizeProp.remove('test_acc')
    normalizeProp.remove('test1_acc')
    normalizeProp.remove('legal')
    normalizeProp.remove('legal1')
    print("normalizeProp")
    print(normalizeProp)
    divideBy = helperFunctions_CriticFrozen.getMaxNumOfNodesInDataSet(df) - helperFunctions_CriticFrozen.getMinNumOfNodesInDataSet(df)
    multiplyBy = maxNumOfLayers - minNumOfLayers
    #Normalize the data for num_layers, num_node1, num_node2 only
    helperFunctions_CriticFrozen.normalizeTwice(df_grid,normalizeProp,divideBy, minNumOfLayers, maxNumOfLayers)
    #helperFunctions_CriticFrozen.normalize(df1,normalizeProp,divideBy, minNumOfLayers, maxNumOfLayers)
    print("df_grid.head() after normalized data")
    print(df_grid.head())
    origProperties = list(df_grid.columns.values)
    trueProperties = list(df_grid.columns.values)
    # Now save two dataframes one for final test which is y (num_layers_N  num_node1_N  num_node2_N, test_acc)
    # and another for X (num_layers_N, num_node1_N, num_node2_N,,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10)

    # remove properties 'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'
    #  and 'num_layers', 'num_node1', 'num_node2'
    randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.removeOneHotVectorTwice(trueProperties)
    trueProperties.remove('num_layers')
    trueProperties.remove('num_node1')
    trueProperties.remove('num_node2')
    trueProperties.remove('num1_layers')
    trueProperties.remove('num1_node1')
    trueProperties.remove('num1_node2')
    #trueProperties.remove('test_acc')
    print("trueProperties")
    print(trueProperties)
    #re-arramge the column order

    # trueProperties will have num_layers_N  num_node1_N  num_node2_N  test_acc  legal num1_layers_N  num1_node1_N  num1_node2_N  test1_acc  legal1
    y = df_grid[['num_layers_N', 'num_node1_N', 'num_node2_N', 'test_acc', 'legal', 'num1_layers_N', 'num1_node1_N', 'num1_node2_N','test1_acc', 'legal1']]

    print("y_true.head() after normalized data")
    print(y.head())
    
    xOrigProperties = list(df_grid.columns.values)
    xOrigProperties.remove('num_layers')
    xOrigProperties.remove('num_node1')
    xOrigProperties.remove('num_node2')
    xOrigProperties.remove('test_acc') 
    xOrigProperties.remove('legal')
    xOrigProperties.remove('num1_layers')
    xOrigProperties.remove('num1_node1')
    xOrigProperties.remove('num1_node2')
    xOrigProperties.remove('test1_acc') 
    xOrigProperties.remove('legal1')
    xProperties = xOrigProperties
    print("xProperties")
    print(xProperties) # 
    X = df_grid[['NoOfAttributes','NoOfClasses','DataSetSize','AttributeType','EntropyLabel','AvgEntrophyFeatures',
                'AvgCorelationBetFeatures','2_6_8_TrainingAccuracy','2_6_8_TestAccuracy','1_5_0_TrainingAccuracy','1_5_0_TestAccuracy',
                '2_6_20_TrainingAccuracy','2_6_20_TestAccuracy', 'num_layers_N', 'num_node1_N', 'num_node2_N',
                'NoOfAttributes1','NoOfClasses1','DataSetSize1','AttributeType1','EntropyLabel1','AvgEntrophyFeatures1',
                'AvgCorelationBetFeatures1','2_6_8_TrainingAccuracy1','2_6_8_TestAccuracy1','1_5_0_TrainingAccuracy1','1_5_0_TestAccuracy1',
                '2_6_20_TrainingAccuracy1','2_6_20_TestAccuracy1', 'num1_layers_N', 'num1_node1_N', 'num1_node2_N']]
    print("X.head() with only num_layers_N num_node1_N num_node2_N oneHotVector num1_layers_N num1_node1_N num1_node2_N oneHotVector")
    print(X.head())
    
    # Divide the data into test and train sets
    a = random.randint(1,50)
    print ("random_state a")
    print (a)
    
    X_train = X
    y_train = y
    test_nn = X_train
    predictions  = model.predict(test_nn)
    resultEval = pd.DataFrame(predictions)
    #Accuracy Legal  no of layers node1 node2
    resultEval.to_csv(current_dir+'nn_architectures_Ecoli_Test_Results.csv', sep='\t', encoding='utf-8', index=False)
    print(predictions)
    #drived_acc = predictions[:,:1]
    #legal = predictions
    # list_truth = y_train.to_numpy()


    # #Take batch of 50 for testing decoder_Accuracy
    # legal_50 = legal[0:50]
    # list_truth_50 = list_truth[0:50]
    # list_truth_50T = tf.convert_to_tensor(list_truth_50)
    # list_truth_50T =  tf.cast(list_truth_50T, tf.float32)
    # list_pred_50T = tf.convert_to_tensor(legal_50)
    # list_pred_50T =  tf.cast(list_pred_50T, tf.float32)
    # decoder_accuracy_legal(list_truth_50T,list_pred_50T)

    # list_truthT = tf.convert_to_tensor(list_truth)
    # list_truthT =  tf.cast(list_truthT, tf.float64)
    # list_predT = tf.convert_to_tensor(legal)
    # list_predT =  tf.cast(list_predT, tf.float64)
    # bce = binaryCrossEntrphy(list_truthT,list_predT)
    # bce_value = tf.keras.backend.get_value(K.mean(bce))
    # diff = K.equal(K.round(list_truthT[:,1:2] - list_predT[:,1:2]), 0) 
    # diff_v = tf.keras.backend.get_value(K.mean(K.cast(diff,tf.float32)) )

    #2,8,4
    test_nn = pd.DataFrame({'F1': [0], 'F2': [1],'F3': [0], 'F4': [0],'F5': [1],
                'F6': [0],'F7': [0], 'F8': [0], 'F9': [0],'F10': [0], 'num_layers_N': [0.6666666],
                'num_node1_N': [0.33333],'num_node2_N': [0.1666666]}, index=['501'])   
    # predictions  = model.predict(test_nn)
    # accuracy = predictions[0][0]
    # legal = predictions[0][1]
    # num_layers = predictions[0][2]
    # node_1 = predictions[0][3]
    # node_2 = predictions[0][4]
    # emb_x = predictions[0][5]
    # emb_y = predictions[0][6]
    # emb_array = [emb_x,emb_y]
    # emb_array = np.array(emb_array)

    # Get predictions from encoder layer for embeddings
    emb = getEmbeddingLayerOutput(model, test_nn)
    print (emb[0][0])
    #split = Lambda(lambda x: tf.split(x,[10,3],axis=1))(test_nn)
    #emb  = encoder_model.predict(split[1])
    #emb[0][0][0] = 3.9227753 
    #emb[0][0][1] = 14.157438
    firstEmb = emb[0][0]
    firstEmbT =   tf.convert_to_tensor(firstEmb, dtype=tf.float32)
    e = pd.DataFrame(emb[0])
    #Accuracy Legal  no of layers node1 node2
    #e.to_csv(current_dir+'embeddings.csv', sep='\t', encoding='utf-8', index=False)
    #embT = tf.convert_to_tensor(emb[0], dtype=tf.float32)
    #  # Get predictions from decder layer for embeddings
    #decode = decoder_model.predict(firstEmbT)  wrong way
    #d = pd.DataFrame(decode)
    #Accuracy Legal  no of layers node1 node2
    #d.to_csv(current_dir+'decoded.csv', sep='\t', encoding='utf-8', index=False)
    decode = getDecoderLayerOutput(decoder_model, firstEmbT)
    layers = decode[0][0][0]
    numLayers = round(layers*3)
    num_node1 = decode[0][0][1] * 24.0
    num_node2 = decode[0][0][2] * 24.0
    emb[0][0][0]=13.26375
    emb[0][0][1]=36.854244
    xy_value = np.append(emb[0][0][0], emb[0][0][1])
    xy_valueT = tf.convert_to_tensor(xy_value, dtype=tf.float32)
    state = xy_valueT
    oneHot = np.array([0., 1., 0., 0., 1., 0., 0., 0., 0., 0.])
    oneHotT = tf.convert_to_tensor(oneHot, dtype=tf.float32)
    state_T = tf.concat([state, oneHotT], axis=0)
    state = tf.keras.backend.get_value(state_T)
    updated_Combined_arr = np.array([list(state)])
    list_combined = []
    list_combined.append(updated_Combined_arr)
 
    #current_accuracy = current_accuracy
    accuracy_prev_state = accuracy_model.predict(list_combined)




    '''
    accuracy_output = getAccuracyLayerOutput(accuracy_model, firstEmbT)
    accuracy = accuracy_output[0][0]
    legal = accuracy_output[0][1]

    oneHot = np.array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
    oneHot = tf.convert_to_tensor(oneHot, dtype=tf.float32)
    concat_layer = Concatenate()([firstEmbT,oneHot])
    concat_layer = tf.convert_to_tensor([concat_layer], dtype=tf.float32)
    '''
    
    num_layers = decode[0][0]
    num_node1 = decode[0][1]
    num_node2 = decode[0][2]
    test_nn1 = pd.DataFrame({'F1': [0], 'F2': [1],'F3': [0], 'F4': [0],'F5': [1],
                    'F6': [0],'F7': [0], 'F8': [0], 'F9': [0],'F10': [0], 'num_layers_N': [num_layers],
                    'num_node1_N': [num_node1],'num_node2_N': [num_node2]})            
    predictions1  = model.predict(test_nn1)
    return  

def generateEmbeddings(grid_DataSet, model,current_dir,featureVectorMinMax):
    
    df_grid = pd.read_csv(grid_DataSet)
    #feature_df = pd.read_csv(featureVectorMinMax)
    normalizeProp = list(df_grid.columns.values)
    # remove 'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'
    randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.removeOneHotVector(normalizeProp)
    #Also remove 'test_acc' so that we dont normalize the accuracy
    normalizeProp.remove('test_acc')
    normalizeProp.remove('legal')
    print("normalizeProp")
    print(normalizeProp)
    #divideBy = helperFunctions_CriticFrozen.getMaxNumOfNodesInDataSet(df_grid) - helperFunctions_CriticFrozen.getMinNumOfNodesInDataSet(df_grid)
    divideBy = 24
    #Normalize the data for num_layers, num_node1, num_node2 only
    helperFunctions_CriticFrozen.normalize(df_grid,normalizeProp,divideBy, 0.0, 3.0)
    xOrigProperties = list(df_grid.columns.values)
    xOrigProperties.remove('num_layers')
    xOrigProperties.remove('num_node1')
    xOrigProperties.remove('num_node2')
    xOrigProperties.remove('test_acc') 
    xOrigProperties.remove('legal')
    xProperties = xOrigProperties
    print("xProperties")
    print(xProperties)
    X = df_grid[xProperties].copy()
    print("X.head() with only num_layers_N  num_node1_N  num_node2_N and one hot vector")
    print(X.head())

    featureProperties = list(X.columns.values)
    featureProperties.remove('num_layers_N')
    featureProperties.remove('num_node1_N')
    featureProperties.remove('num_node2_N')
    # No need to normalize accuracies
    featureProperties.remove('2_6_8_TrainingAccuracy')
    featureProperties.remove('2_6_8_TestAccuracy')
    featureProperties.remove('1_5_0_TrainingAccuracy')
    featureProperties.remove('1_5_0_TestAccuracy')
    featureProperties.remove('2_6_20_TrainingAccuracy')
    featureProperties.remove('2_6_20_TestAccuracy')


    # Normalize X feature vectors
    helperFunctions_CriticFrozen.normalizeFeatureVectoreTwice(X,featureProperties,featureVectorMinMax)
    print (list(X.columns.values))
    X = X [['NoOfAttributes','NoOfClasses','DataSetSize','AttributeType','EntropyLabel','AvgEntrophyFeatures',
                'AvgCorelationBetFeatures','2_6_8_TrainingAccuracy','2_6_8_TestAccuracy','1_5_0_TrainingAccuracy','1_5_0_TestAccuracy',
                '2_6_20_TrainingAccuracy','2_6_20_TestAccuracy', 'num_layers_N', 'num_node1_N', 'num_node2_N']]

    print (list(X.columns.values))
    print("X.head() Normalized")
    print(X.head())
    X_train = X

    test_nn = X_train
    # Get predictions from encoder layer for embeddings
    emb = getEmbeddingLayerOutput(model, test_nn) 
    e = pd.DataFrame(emb[0],columns=['emb_x', 'emb_y'])
    #concatinate with Normalized Feature Vector
    split = Lambda(lambda x: tf.split(x,[13,3],axis=1))(test_nn)
    
    hotVector = tf.cast((split[0]), tf.float32)
    hotVector = tf.keras.backend.get_value(hotVector)
    h = pd.DataFrame(hotVector,columns=['NoOfAttributes','NoOfClasses','DataSetSize','AttributeType','EntropyLabel','AvgEntrophyFeatures',
                                        'AvgCorelationBetFeatures','2_6_8_TrainingAccuracy','2_6_8_TestAccuracy','1_5_0_TrainingAccuracy','1_5_0_TestAccuracy',
                                        '2_6_20_TrainingAccuracy','2_6_20_TestAccuracy'])
    df_merged = pd.concat([e, h], axis=1)

    
    #Accuracy Legal  no of layers node1 node2
    #df_merged.to_csv(current_dir+'embeddings_nn_architectures_Ecoli_Test1.csv', sep=',', encoding='utf-8', index=False)
    df_merged.to_csv(current_dir+'embeddings_9NN_VectorTrainingFileCombined1.csv', sep=',', encoding='utf-8', index=False)
    #df_merged.to_csv(current_dir+'embeddings_unique_test_decoder.csv', sep=',', encoding='utf-8', index=False)




def generateEmbPredictions(emd_DataSet, accuracy_model,model, current_dir):
    edf = pd.read_csv(emd_DataSet)
    listOfEmbLegal = []
    for index, row in edf.iterrows():
        emb_x = row["emb_x"]
        emb_y = row["emb_y"]
        xy_value = np.append(emb_x, emb_y)
        xy_valueT = tf.convert_to_tensor(xy_value, dtype=tf.float32)
        decode = getDecoderLayerOutput(model, xy_valueT)
        numLayers = decode[0][0][0] *3
        num_node1 = decode[0][0][1] * 24.0
        num_node2 = decode[0][0][2] * 24.0
        numLayers_r = round(decode[0][0][0] *3 )
        num_node1_r = round(decode[0][0][1] * 24.0)
        num_node2_r = round(decode[0][0][2] * 24.0)
        # state = xy_valueT
        # oneHot = np.array([0., 1., 0., 0., 1., 0., 0., 0., 0., 0.])
        # oneHotT = tf.convert_to_tensor(oneHot, dtype=tf.float32)
        # state_T = tf.concat([state, oneHotT], axis=0)
        # state = tf.keras.backend.get_value(state_T)
        # updated_Combined_arr = np.array([list(state)])
        #Split the row
        list_combined = []
        list_combined.append(row)
        row = tf.convert_to_tensor(list_combined, dtype=tf.float32)

        splitRow = Lambda(lambda x: tf.split(x,[2,13],axis=1))(row)
        encoder_row_Hot = vector_drop_layer1(vector_layer1(vector_noise1(splitRow[1])))
        concat_layer = Concatenate()([splitRow[0],encoder_row_Hot])
        # updated_Combined_arr = np.array([list(row)])
        # list_combined = []
        # list_combined.append(updated_Combined_arr)
        accOutput = getAccuracyLayerOutput(concat_layer,accuracy_model)
        setOfEmb = (emb_x,emb_y,numLayers,num_node1,num_node2,numLayers_r,num_node1_r,num_node2_r,accOutput[0][0][0],round(accOutput[0][0][1]))
        listOfEmbLegal.append(setOfEmb)
    # write listOfgridData_DrivenAcc in a csv file
    item_length = len(listOfEmbLegal)
    print (item_length)
    with open(current_dir+'embeddings_9NN_VectorTrainingFileCombined_Predictions1.csv', 'w') as emdFile:
    #with open(current_dir+'embeddings_nn_architectures_Ecoli_Predictions1.csv', 'w') as emdFile:
    #with open(current_dir+'embeddings_unique_test_decoder_predictions.csv', 'w') as emdFile:
        file_writer = csv.writer(emdFile)
        for i in range(item_length):
            file_writer.writerow([str(element) for element in listOfEmbLegal[i]])
        return 



def generatePredictionsForEmbeddings(grid_DataSet, model,current_dir):
    df_grid = pd.read_csv(grid_DataSet)
    #df_grid.columns = ['emb_x', 'emb_y', 'current_accuracy']
    #print(df_grid.head())
    listOfAccuracy = []
    list_Legal = []

    for index, row in df_grid.iterrows():
        num_layers = row["num_layers"]
        num_layers = num_layers/3.0

        num_node1 = row["num_node1"]
        num_node1 = num_node1/24.0

        num_node2 = row["num_node2"]
        num_node2 = num_node2/24.0
    

        test_nn = pd.DataFrame({'F1': [0], 'F2': [1],'F3': [0], 'F4': [0],'F5': [1],
                    'F6': [0],'F7': [0], 'F8': [0], 'F9': [0],'F10': [0], 'num_layers_N': [num_layers],
                    'num_node1_N': [num_node1],'num_node2_N': [num_node2]}, index=['50112'])
        print(test_nn.head())
        # Generate predictions for samples
        predictions  = model.predict(test_nn)
        print(predictions)
        drived_acc = predictions[0][0]
        legal = predictions[0][1] 

        listOfAccuracy.append(drived_acc)
        list_Legal.append(legal)
    with open(current_dir+'listOfAccuracy.csv','w',newline='') as file:
        write=csv.writer(file)
        for num in listOfAccuracy:
            write.writerow([num])
    with open(current_dir+'list_Legal.csv','w',newline='') as file:
        write=csv.writer(file)
        for num in list_Legal:
            write.writerow([num])

 

# #Euclidean distance between two embeddings
# def lambda_function(concat_embeddings):
#     #split_emb = Lambda(lambda x: tf.split(x,[2,2],axis=1))(concat_embeddings)
#     distance =  K.sqrt(K.sum(K.square(concat_embeddings[:,0:2] - concat_embeddings[:,2:4]),axis=-1))
#     distance = tf.reshape(distance,[-1,1])
#     #make sure we dont have infinite distance + some constant to allow reasonable emb space
#     distanceNormalize = tf.keras.activations.tanh(distance * 0.1)  
#     return distanceNormalize
#   #return K.sqrt(K.sum(K.square(encoder_OUT - encoder_OUT1), axis=-1))

# #Euclidean distance between two embeddings
def lambda_function(concat_embeddings):
    sumSquared = K.sum(K.square(concat_embeddings[:,0:2] - concat_embeddings[:,2:4]), axis=1,keepdims=True)
    distance = K.sqrt(K.maximum(sumSquared, K.epsilon()))
    distanceNormalize = tf.keras.activations.tanh(distance * 0.1) 
    return distanceNormalize
  #return K.sqrt(K.sum(K.square(encoder_OUT - encoder_OUT1), axis=-1))


def makeRandom100pairs(df,df1):
# 10 elements in DF
# 1 from df and random 100 of df1 = df3    
# twice colums 
# df3 has 100 elements pair
# df3 to the encode-decoder-accuracy networks and take different sets while training them
# make pairs for each df make 100 random pairs with df1
    df3 = pd.DataFrame(columns=['num_layers','num_node1','num_node2','NoOfAttributes','NoOfClasses','DataSetSize','AttributeType','EntropyLabel','AvgEntrophyFeatures',
                       	'AvgCorelationBetFeatures','2_6_8_TrainingAccuracy','2_6_8_TestAccuracy','1_5_0_TrainingAccuracy','1_5_0_TestAccuracy','2_6_20_TrainingAccuracy','2_6_20_TestAccuracy','test_acc','legal',
                         'num1_layers','num1_node1','num1_node2','NoOfAttributes1','NoOfClasses1','DataSetSize1','AttributeType1','EntropyLabel1','AvgEntrophyFeatures1',
                       	'AvgCorelationBetFeatures1','2_6_8_TrainingAccuracy1','2_6_8_TestAccuracy1','1_5_0_TrainingAccuracy1','1_5_0_TestAccuracy1','2_6_20_TrainingAccuracy1','2_6_20_TestAccuracy1','test1_acc','legal1'])
    for index, row in df.iterrows():
        x = 0
        #for number in range(100):
        for number in range(100):
            y = random.randint(0, 899)
            while x == y: # we dont want same random number 
                 y = random.randint(0, 899)
            x = y     
            row1 = df1.loc[y]
            df3 = df3.append({'num_layers':float(row["num_layers"]),'num_node1':float(row["num_node1"]),'num_node2':float(row["num_node2"]),
                              'NoOfAttributes': float(row["NoOfAttributes"]), 'NoOfClasses': float(row["NoOfClasses"]),'DataSetSize': float(row["DataSetSize"]), 'AttributeType': float(row["AttributeType"]),'EntropyLabel': float(row["EntropyLabel"]),
                              'AvgEntrophyFeatures': float(row["AvgEntrophyFeatures"]),'AvgCorelationBetFeatures': float(row["AvgCorelationBetFeatures"]), '2_6_8_TrainingAccuracy': float(row["2_6_8_TrainingAccuracy"]), 
                              '2_6_8_TestAccuracy': float(row["2_6_8_TestAccuracy"]),'1_5_0_TrainingAccuracy': float(row["1_5_0_TrainingAccuracy"]),'1_5_0_TestAccuracy': float(row["1_5_0_TestAccuracy"]),
                              '2_6_20_TrainingAccuracy': float(row["2_6_20_TrainingAccuracy"]),'2_6_20_TestAccuracy': float(row["2_6_20_TestAccuracy"]), 
                              'test_acc':float(row["test_acc"]),'legal':float(row["legal"]),
                              'num1_layers':float(row1["num_layers"]),'num1_node1':float(row1["num_node1"]),'num1_node2':float(row1["num_node2"]),
                              'NoOfAttributes1': float(row1["NoOfAttributes"]), 'NoOfClasses1': float(row1["NoOfClasses"]),'DataSetSize1': float(row1["DataSetSize"]), 'AttributeType1': float(row1["AttributeType"]),'EntropyLabel1': float(row1["EntropyLabel"]),
                              'AvgEntrophyFeatures1':float(row1["AvgEntrophyFeatures"]),'AvgCorelationBetFeatures1':float(row1["AvgCorelationBetFeatures"]), '2_6_8_TrainingAccuracy1':float(row1["2_6_8_TrainingAccuracy"]), 
                              '2_6_8_TestAccuracy1':float(row1["2_6_8_TestAccuracy"]),'1_5_0_TrainingAccuracy1':float(row1["1_5_0_TrainingAccuracy"]),'1_5_0_TestAccuracy1':float(row1["1_5_0_TestAccuracy"]),
                              '2_6_20_TrainingAccuracy1':float(row1["2_6_20_TrainingAccuracy"]),'2_6_20_TestAccuracy1':float(row1["2_6_20_TestAccuracy"]), 
                              'test1_acc':float(row1["test_acc"]),'legal1':float(row1["legal"])},ignore_index = True)
            print(df3)
    return df3




#
#   CODE FOR NETWORK ONE without RL
#   
# Dense Layers nodes for encoder and decoder
maxNumOfLayers = 3.0
minNumOfLayers = 0.0 
latent_dim1 = 40
latent_dim11 = 40
latent_dim111 = 40
latent_dim1111 = 40
divideBy = 24.0
multiplyBy = 3.0
#os.chdir("/home/bvadhera/huber/")
cwd = os.getcwd()
#current_dir  = "/home/bvadhera/huber/"
current_dir  = cwd +"/"


print (current_dir)


inputNN_Architectures_DataSet = current_dir+"9NN_VectorTrainingFileCombined.csv"
minmaxAccuracyFile = current_dir+"secondNetwork38234MinMax.csv"
print (inputNN_Architectures_DataSet)
writeFile = current_dir + "results_" + "9NN_VectorTrainingFileCombined_paired.csv"
myFile = open(writeFile, 'w')

##  TODO - CHange the headings as per the results
header = ['training_loss','val_loss', 'custom_loss', 'val_custom_loss','decoder_accuracy_legal',  'val_decoder_accuracy_legal', 'decoder_loss',   'val_decoder_loss',
         'binaryCrossEntrphy', 'val_binaryCrossEntrphy','legal_Network_Pred',  'val_legal_Network_Pred','mean_sqe_pred',  'val_mean_sqe_pred', 
         'acc_loss','val_acc_loss','legal_loss','val_legal_loss','distance_loss', 'val_distance_loss' ] 


writer = csv.DictWriter(myFile, fieldnames=header) 
writer.writeheader()
print("writeFile")
print(writeFile)


# read file to read min and max
fileMinMax = open(minmaxAccuracyFile, 'r')
MinMaxLine = fileMinMax.readline()
fileMinMax.close()
featureVectorMinMax = current_dir+"featureVectorMinMax.csv"
#split string by ,
chunks = MinMaxLine.split(',')
max_accuracy_before_norm = float(chunks[0])
min_accuracy_before_norm = float(chunks[1])
maxNumOfLayers = float(chunks[2])
minNumOfLayers = float(chunks[3])
# Open a csv reader called DictReader
# iterate over each line as a ordered dictionary and print only few column by column name
df = pd.read_csv(inputNN_Architectures_DataSet)
df1 = shuffle(df)
# df3 = makeRandom100pairs(df,df1)
# df3.to_csv(current_dir+"9NN_VectorTrainingFileCombined_paired.csv",index=False, encoding='utf-8')
df3 = pd.read_csv(current_dir+"9NN_VectorTrainingFileCombined_paired.csv")
#df3 = pd.read_csv(current_dir+"mixedDataFramesSample.csv")
#df.to_csv(current_dir+"duplicateDF.csv", sep='\t', encoding='utf-8')
#print(df.head())
print(df3.head())
normalizeProp = list(df3.columns.values)
# remove 'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'
randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.removeOneHotVectorTwice(normalizeProp)
#Also remove 'test_acc' so that we dont normalize the accuracy
normalizeProp.remove('test_acc')
normalizeProp.remove('test1_acc')
normalizeProp.remove('legal')
normalizeProp.remove('legal1')
print("normalizeProp")
print(normalizeProp)
divideBy = helperFunctions_CriticFrozen.getMaxNumOfNodesInDataSet(df) - helperFunctions_CriticFrozen.getMinNumOfNodesInDataSet(df)
multiplyBy = maxNumOfLayers - minNumOfLayers
#Normalize the data for num_layers, num_node1, num_node2 only
helperFunctions_CriticFrozen.normalizeTwice(df3,normalizeProp,divideBy, minNumOfLayers, maxNumOfLayers)
#helperFunctions_CriticFrozen.normalize(df1,normalizeProp,divideBy, minNumOfLayers, maxNumOfLayers)
print("df3.head() after normalized data")
print(df3.head())
origProperties = list(df3.columns.values)
trueProperties = list(df3.columns.values)
# Now save two dataframes one for final test which is y (num_layers_N  num_node1_N  num_node2_N, test_acc)
# and another for X (num_layers_N, num_node1_N, num_node2_N,,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10)

# remove properties 'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'
#  and 'num_layers', 'num_node1', 'num_node2'
randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.removeOneHotVectorTwice(trueProperties)
trueProperties.remove('num_layers')
trueProperties.remove('num_node1')
trueProperties.remove('num_node2')
trueProperties.remove('num1_layers')
trueProperties.remove('num1_node1')
trueProperties.remove('num1_node2')
#trueProperties.remove('test_acc')
print("trueProperties")
print(trueProperties)
#re-arramge the column order

# trueProperties will have num_layers_N  num_node1_N  num_node2_N  test_acc  legal num1_layers_N  num1_node1_N  num1_node2_N  test1_acc  legal1
y = df3[['num_layers_N', 'num_node1_N', 'num_node2_N', 'test_acc', 'legal', 'num1_layers_N', 'num1_node1_N', 'num1_node2_N','test1_acc', 'legal1']]

print("y_true.head() after normalized data")
print(y.head())
 
xOrigProperties = list(df3.columns.values)
xOrigProperties.remove('num_layers')
xOrigProperties.remove('num_node1')
xOrigProperties.remove('num_node2')
xOrigProperties.remove('test_acc') 
xOrigProperties.remove('legal')
xOrigProperties.remove('num1_layers')
xOrigProperties.remove('num1_node1')
xOrigProperties.remove('num1_node2')
xOrigProperties.remove('test1_acc') 
xOrigProperties.remove('legal1')
xProperties = xOrigProperties
print("xProperties")
print(xProperties) # 

X = df3[['NoOfAttributes','NoOfClasses','DataSetSize','AttributeType','EntropyLabel','AvgEntrophyFeatures',
            'AvgCorelationBetFeatures','2_6_8_TrainingAccuracy','2_6_8_TestAccuracy','1_5_0_TrainingAccuracy','1_5_0_TestAccuracy',
            '2_6_20_TrainingAccuracy','2_6_20_TestAccuracy', 'num_layers_N', 'num_node1_N', 'num_node2_N',
            'NoOfAttributes1','NoOfClasses1','DataSetSize1','AttributeType1','EntropyLabel1','AvgEntrophyFeatures1',
            'AvgCorelationBetFeatures1','2_6_8_TrainingAccuracy1','2_6_8_TestAccuracy1','1_5_0_TrainingAccuracy1','1_5_0_TestAccuracy1',
            '2_6_20_TrainingAccuracy1','2_6_20_TestAccuracy1', 'num1_layers_N', 'num1_node1_N', 'num1_node2_N']].copy()

print("X.head() with only num_layers_N num_node1_N num_node2_N oneHotVector num1_layers_N num1_node1_N num1_node2_N oneHotVector")
print(X.head())

featureProperties = list(X.columns.values)
featureProperties.remove('num_layers_N')
featureProperties.remove('num_node1_N')
featureProperties.remove('num_node2_N')
featureProperties.remove('num1_layers_N')
featureProperties.remove('num1_node1_N')
featureProperties.remove('num1_node2_N')
print(featureProperties)

acc_featureProperties = featureProperties.copy()
acc_featureProperties.remove('NoOfAttributes')
acc_featureProperties.remove('NoOfClasses')
acc_featureProperties.remove('DataSetSize')
acc_featureProperties.remove('AttributeType')
acc_featureProperties.remove('EntropyLabel')
acc_featureProperties.remove('AvgEntrophyFeatures')
acc_featureProperties.remove('AvgCorelationBetFeatures')
acc_featureProperties.remove('NoOfAttributes1')
acc_featureProperties.remove('NoOfClasses1')
acc_featureProperties.remove('DataSetSize1')
acc_featureProperties.remove('AttributeType1')
acc_featureProperties.remove('EntropyLabel1')
acc_featureProperties.remove('AvgEntrophyFeatures1')
acc_featureProperties.remove('AvgCorelationBetFeatures1')

print(featureProperties)
# No need to normalize accuracies
featureProperties.remove('2_6_8_TrainingAccuracy')
featureProperties.remove('2_6_8_TestAccuracy')
featureProperties.remove('1_5_0_TrainingAccuracy')
featureProperties.remove('1_5_0_TestAccuracy')
featureProperties.remove('2_6_20_TrainingAccuracy')
featureProperties.remove('2_6_20_TestAccuracy')
featureProperties.remove('2_6_8_TrainingAccuracy1')
featureProperties.remove('2_6_8_TestAccuracy1')
featureProperties.remove('1_5_0_TrainingAccuracy1')
featureProperties.remove('1_5_0_TestAccuracy1')
featureProperties.remove('2_6_20_TrainingAccuracy1')
featureProperties.remove('2_6_20_TestAccuracy1')
nonAcc_featureProperties = featureProperties

# Normalize X feature vectors only non Accuracy
helperFunctions_CriticFrozen.normalizeFeatureVectoreTwice(X,nonAcc_featureProperties,featureVectorMinMax)
print (list(X.columns.values))
X = X [['NoOfAttributes','NoOfClasses','DataSetSize','AttributeType','EntropyLabel','AvgEntrophyFeatures',
            'AvgCorelationBetFeatures','2_6_8_TrainingAccuracy','2_6_8_TestAccuracy','1_5_0_TrainingAccuracy','1_5_0_TestAccuracy',
            '2_6_20_TrainingAccuracy','2_6_20_TestAccuracy', 'num_layers_N', 'num_node1_N', 'num_node2_N',
            'NoOfAttributes1','NoOfClasses1','DataSetSize1','AttributeType1','EntropyLabel1','AvgEntrophyFeatures1',
            'AvgCorelationBetFeatures1','2_6_8_TrainingAccuracy1','2_6_8_TestAccuracy1','1_5_0_TrainingAccuracy1','1_5_0_TestAccuracy1',
            '2_6_20_TrainingAccuracy1','2_6_20_TestAccuracy1', 'num1_layers_N', 'num1_node1_N', 'num1_node2_N']].copy()



nonAcc_featureProperties1 = ['NoOfAttributes','NoOfClasses','DataSetSize','AttributeType','EntropyLabel','AvgEntrophyFeatures','AvgCorelationBetFeatures']
nonAcc_featureProperties2 = ['NoOfAttributes1','NoOfClasses1','DataSetSize1','AttributeType1','EntropyLabel1','AvgEntrophyFeatures1','AvgCorelationBetFeatures1']
acc_featureProperties1 = ['2_6_8_TrainingAccuracy','2_6_8_TestAccuracy','1_5_0_TrainingAccuracy','1_5_0_TestAccuracy','2_6_20_TrainingAccuracy','2_6_20_TestAccuracy']
acc_featureProperties2 = ['2_6_8_TrainingAccuracy1','2_6_8_TestAccuracy1','1_5_0_TrainingAccuracy1','1_5_0_TestAccuracy1','2_6_20_TrainingAccuracy1','2_6_20_TestAccuracy1']
# Adding STD to feature vectors and multiply them by 0.2 and accuracy features by 0.1 only 
NOISE_STD_VECTOR1 = helperFunctions_CriticFrozen.stdToFeatures(X,nonAcc_featureProperties1,acc_featureProperties1)
NOISE_STD_VECTOR2 = helperFunctions_CriticFrozen.stdToFeatures(X,nonAcc_featureProperties2,acc_featureProperties2)

print("NOISE_STD_VECTOR1")
print(NOISE_STD_VECTOR1)
print("NOISE_STD_VECTOR2")
print(NOISE_STD_VECTOR2)

print (list(X.columns.values))
print("X.head() Normalized")
print(X.head())

# Divide the data into test and train sets
a = random.randint(1,50)
print ("random_state a")
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
print('y_train.type:',type(y_train))   # shape 

# Get Batch size for X_train
batchSize = 500
print ("batchSize")
print (batchSize)   

# Instantiate an optimizer.
learning_rate = 0.01
optimizer = tf.keras.optimizers.Adam( clipnorm=1.0)

#  0100100000 num_layers_N  num_node1_N  num_node2_N  0100100000  num1_layers_N  num1_node1_N  num1_node2_N 
# Create a joint NN with split input layer(s))
input_layer = Input(shape=(32,))
print("input_layer.shape")
print(input_layer.shape)
print("type(input_layer)")
print(type(input_layer) )
split = Lambda(lambda x: tf.split(x,[13,3,13,3],axis=1))(input_layer)
#split1 = Lambda( lambda x: tf.split(x,[11,2],axis=1))(input_layer)
print ('split[0].shape:', split[0].shape)  # shape feature vector
print ('split[1].shape:',split[1].shape)   # shape  num_layers_N  num_node1_N  num_node2_N
print ('split1[1].shape:', split[2].shape)  # shape feature vector
print ('split1[2].shape:',split[3].shape)   # shape  num_layers_N  num_node1_N  num_node2_N

inputs_feature_Conversion_model=Input((13,))
vector_noise1 = tf.keras.layers.GaussianNoise(NOISE_STD_VECTOR1)
vector_layer1 = Dense(6, tf.nn.leaky_relu)
vector_drop_layer1 = tf.keras.layers.Dropout(0.2)
feature_Conversion_model = Model(inputs=inputs_feature_Conversion_model , outputs=[vector_drop_layer1(vector_layer1(vector_noise1(inputs_feature_Conversion_model,training=True)),training=True)])

#feature_Conversion_model(inputs_feature_Conversion_model, training=True)  


vector_noise2 = tf.keras.layers.GaussianNoise(NOISE_STD_VECTOR2)
vector_layer2 = Dense(6, tf.nn.leaky_relu)
vector_drop_layer2 = tf.keras.layers.Dropout(0.2)


'''
# to use them as parellel layers
# Create a joint NN with split input layer(s))
input_layer1 = Input(shape=(26,))
print("input_layer1.shape")
print(input_layer1.shape)
print("type(input_layer1)")
print(type(input_layer1) )
split1 = Lambda(lambda x: tf.split(x,[13,10,3],axis=1))(input_layer1)
#split1 = Lambda( lambda x: tf.split(x,[11,2],axis=1))(input_layer)
print ('split1[1].shape:', split1[1].shape)  # shape one hot vector
print ('split1[2].shape:',split1[2].shape)   # shape  num_layers_N  num_node1_N  num_node2_N
'''
#+++++++++++++++++++++++++++++++++++++
# Building TRACK - 1
#++++++++++++++++++++++++++++++++
# nn_layer goes int encoder  
# Hidden Layers  

inputs_encoder_model=Input((3,))
activation=tf.nn.leaky_relu
encoder_layer_1 = Dense(latent_dim1, activation, input_shape=[3])
encoder_layer_11 = Dense(latent_dim11, activation)
encoder_layer_111 = Dense(latent_dim111, activation)
encoder_layer_1111= Dense(latent_dim1111, activation) 
#output embedding (x,y) Layer
embedding_layer = Dense(int(2), activation=tf.keras.activations.tanh) 


#print ('embedding_layer.shape:', embedding_layer.shape)
encoder_model = Model(inputs=inputs_encoder_model, outputs=[embedding_layer(encoder_layer_1111(encoder_layer_111(encoder_layer_11(encoder_layer_1(inputs_encoder_model)))))])
encoder_model.compile(optimizer=optimizer, loss=None,metrics=None)

print (encoder_model.summary())


inputs_encoder_model1=Input((3,))
#print ('embedding_layer.shape:', embedding_layer.shape)
encoder_model1 = Model(inputs=inputs_encoder_model1, outputs=[embedding_layer(encoder_layer_1111(encoder_layer_111(encoder_layer_11(encoder_layer_1(inputs_encoder_model1)))))])
encoder_model1.compile(optimizer=optimizer, loss=None,metrics=None)

print (encoder_model1.summary())




#++++++++++++++++++++++++++++++++
# embedding_layer goes int decoder  
#Hidden Layers
inputs_decoder_model=Input((2,))
decoder_layer_1111 = Dense(latent_dim1111, activation=tf.nn.leaky_relu) 
decoder_layer_111 = Dense(latent_dim111, activation=tf.nn.leaky_relu)
decoder_layer_11 = Dense(latent_dim11, activation=tf.nn.leaky_relu)
decoder_layer_1 = Dense(latent_dim1, activation=tf.nn.leaky_relu) 

#output decoded (x,y) Layer
decoded_layer = Dense(int(3), activation=tf.nn.sigmoid) ## To DO Activation 0-1
#print ('decoded_layer.shape:', decoded_layer.shape)
#split_ = Lambda(lambda x: tf.split(x,[10,3],axis=1))(X_train)
encoder_OP = embedding_layer (encoder_layer_1111(encoder_layer_111(encoder_layer_11(encoder_layer_1(split[1])))))
decoder_model = Model(inputs=[inputs_decoder_model], outputs=[decoded_layer(decoder_layer_1(decoder_layer_11(decoder_layer_111(decoder_layer_1111(inputs_decoder_model)))))])

 
filepath=current_dir +"acc_encoder_decoder_vector_distance/" #modelFile+"_"+"acc."
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min') #monitor='val_tpr',
callbacks_list_NN = [checkpoint]
#First, call compile to configure the optimizer, loss, and metrics to monitor. 
decoder_model.compile(optimizer=optimizer, loss=decoder_loss, metrics=[decoder_accuracy_legal])
print (decoder_model.summary())

#++++++++++++++++++++++++++++++++
# embedding_layer goes int decoder  
#Hidden Layers


#output decoded (x,y) Layer
inputs_decoder_model1=Input((2,))
#print ('decoded_layer.shape:', decoded_layer.shape)
#split_ = Lambda(lambda x: tf.split(x,[10,3],axis=1))(X_train)
encoder_OP1 = embedding_layer (encoder_layer_1111(encoder_layer_111(encoder_layer_11(encoder_layer_1(split[3])))))
decoder_model1 = Model(inputs=[inputs_decoder_model1], outputs=[decoded_layer(decoder_layer_1(decoder_layer_11(decoder_layer_111(decoder_layer_1111(inputs_decoder_model1)))))])



#learning_rate = 0.01
#optimizer = tf.keras.optimizers.Adam(learning_rate)
#optimizer = tf.keras.optimizers.Adam(learning_rate)
 
filepath1=current_dir +"acc_encoder_decoder_vector_distance1/" #modelFile+"_"+"acc."
checkpoint1 = ModelCheckpoint(filepath1, monitor='val_loss', verbose=1, save_best_only=True, mode='min') #monitor='val_tpr',
callbacks_list_NN1 = [checkpoint1]
#First, call compile to configure the optimizer, loss, and metrics to monitor. 
decoder_model1.compile(optimizer=optimizer, loss=decoder_loss, metrics=[decoder_accuracy_legal])
print (decoder_model1.summary())
  



inputs_accuracy_model=Input((8,))

splitAccuracy = Lambda(lambda x: tf.split(x,[2,6],axis=1))(inputs_accuracy_model)

hiddenP_layer_1b = Dense(int(40), activation=tf.nn.leaky_relu, input_shape=[8]) 
hiddenP_layer_11b = Dense(int(40), activation=tf.nn.leaky_relu)  
hiddenP_layer_111b = Dense(int(40), activation=tf.nn.leaky_relu)
hiddenP_layer_1111b = Dense(int(40), activation=tf.nn.leaky_relu) 
hiddenP_layer_11111b = Dense(int(40), activation=tf.nn.leaky_relu)
accuracy_output = Dense(int(1), activation='sigmoid')



hiddenP_layer_1a = Dense(int(20), activation=tf.nn.leaky_relu, input_shape=[2]) 
hiddenP_layer_11a = Dense(int(20), activation=tf.nn.leaky_relu) 
#hiddenP_layer_111a = Dense(int(20), activation=tf.nn.leaky_relu)
hiddenP_layer_1111a = Dense(int(20), activation=tf.nn.leaky_relu) 
#hiddenP_layer_1111ab = Dense(int(2), activation=tf.nn.leaky_relu) - to Acc 1
hiddenP_layer_11111a = Dense(int(20), activation=tf.nn.leaky_relu) 
legal_output = Dense(int(1), activation='sigmoid')
#accuracy_layer = Concatenate(name='accuracy_network')([accuracy_output,legal_output])

acc_output1 =  hiddenP_layer_1111b(hiddenP_layer_111b(hiddenP_layer_11b(hiddenP_layer_1b(inputs_accuracy_model))))
l_layer = hiddenP_layer_11111a(hiddenP_layer_1111a(hiddenP_layer_11a(hiddenP_layer_1a(splitAccuracy[0]))))
l_output = legal_output(l_layer)
 
merged_layer = Concatenate(name='merged_network')([acc_output1,l_layer,l_output])
acc_output = accuracy_output(hiddenP_layer_11111b(merged_layer))
accuracy_layer = Concatenate(name='accuracy_network')([acc_output,l_output])



accuracy_model = Model(inputs=inputs_accuracy_model, outputs=accuracy_layer)
accuracy_model.compile(optimizer=optimizer, loss={'accuracy_network': custom_loss_accuracy},
                     metrics={'accuracy_network': [ mean_sqe_pred, binaryCrossEntrphy]}) #, run_eagerly=True)

filepath_acc=current_dir +"acc_acc_prob2_min_distance/" #modelFile+"_"+"acc"
#accuracy_model.load_weights(filepath_acc + "weights")

print(accuracy_model.summary())
checkpoint2 = ModelCheckpoint(filepath_acc, monitor='val_loss', verbose=1, save_best_only=True, mode='min') #monitor='val_tpr',
callbacks_list_NN_Acc = [checkpoint2]

# Connecting embedding layer
encoder_OUT = embedding_layer(encoder_layer_1111(encoder_layer_111(encoder_layer_11(encoder_layer_1(split[1])))))
encoder_Hot = vector_drop_layer1(vector_layer1(vector_noise1(split[0])))
concat_layer = Concatenate()([encoder_OUT,encoder_Hot])
# Merge predict_layer (predicted Accuracy) back with decoded_layer

acc_output1 =  hiddenP_layer_1111b(hiddenP_layer_111b(hiddenP_layer_11b(hiddenP_layer_1b(concat_layer))))
l_layer = hiddenP_layer_11111a(hiddenP_layer_1111a(hiddenP_layer_11a(hiddenP_layer_1a(encoder_OUT))))
l_output = legal_output(l_layer)
merged_layer = Concatenate(name='merged_network')([acc_output1,l_layer,l_output])
acc_output = accuracy_output(hiddenP_layer_11111b(merged_layer))
accuracy_layer = Concatenate(name='accuracy_network')([acc_output,l_output])


predict_output = Concatenate(name='network_with_accuracy') ([accuracy_layer,
                                    decoded_layer(decoder_layer_1(decoder_layer_11(decoder_layer_111(decoder_layer_1111( encoder_OUT ))))) ] )
#predict_output = Concatenate(name='network_with_accuracy') ([predict_output1,concat_layer] ) 

#------------------------------------
#duplicate Accuracy
  
inputs_accuracy_model1=Input((8,))
splitAccuracy1 = Lambda(lambda x: tf.split(x,[2,6],axis=1))(inputs_accuracy_model1)

acc_output22 =  hiddenP_layer_1111b(hiddenP_layer_111b(hiddenP_layer_11b(hiddenP_layer_1b(inputs_accuracy_model1))))
l_layer1 = hiddenP_layer_11111a(hiddenP_layer_1111a(hiddenP_layer_11a(hiddenP_layer_1a(splitAccuracy1[0]))))
l_output1 = legal_output(l_layer1)
merged_layer1 = Concatenate(name='merged_network1')([acc_output22,l_layer1,l_output1])
acc_output2 = accuracy_output(hiddenP_layer_11111b(merged_layer1))
accuracy_layer1 = Concatenate(name='accuracy_network1')([acc_output2,l_output1])


accuracy_model1 = Model(inputs=inputs_accuracy_model1, outputs=accuracy_layer1)
accuracy_model1.compile(optimizer=optimizer, loss={'accuracy_network': custom_loss_accuracy},
                     metrics={'accuracy_network': [ mean_sqe_pred, binaryCrossEntrphy]}) #, run_eagerly=True)

filepath_acc1=current_dir +"acc_acc_prob2_min_distance1/" #modelFile+"_"+"acc"
print(accuracy_model1.summary())
checkpoint3 = ModelCheckpoint(filepath_acc1, monitor='val_loss', verbose=1, save_best_only=True, mode='min') #monitor='val_tpr',
callbacks_list_NN_Acc1 = [checkpoint3]
 
# Connecting embedding layer
encoder_OUT1 = embedding_layer(encoder_layer_1111(encoder_layer_111(encoder_layer_11(encoder_layer_1(split[3])))))

encoder_Hot1 = vector_drop_layer1(vector_layer1(vector_noise1(split[2])))
concat_layer1 = Concatenate()([encoder_OUT1,encoder_Hot1])
# Merge predict_layer (predicted Accuracy) back with decoded_layer

#common = hiddenP_layer_111(hiddenP_layer_11(hiddenP_layer_1(concat_layer)))

acc_output22 =  hiddenP_layer_1111b(hiddenP_layer_111b(hiddenP_layer_11b(hiddenP_layer_1b(concat_layer1))))
l_layer1 = hiddenP_layer_11111a(hiddenP_layer_1111a(hiddenP_layer_11a(hiddenP_layer_1a(encoder_OUT1))))
l_output1 = legal_output(l_layer1)
merged_layer1 = Concatenate(name='merged_network1')([acc_output22,l_layer1,l_output1])
acc_output2 = accuracy_output(hiddenP_layer_11111b(merged_layer1))
accuracy_layer1 = Concatenate(name='accuracy_network1')([acc_output2,l_output1])


predict_output1 = Concatenate(name='network_with_accuracy1') ([accuracy_layer1,
                                    decoded_layer(decoder_layer_1(decoder_layer_11(decoder_layer_111(decoder_layer_1111( encoder_OUT1 ))))) ] )
#predict_output111 = Concatenate(name='network_with_accuracy') ([predict_output11,concat_layer1] ) 
concat_embeddings = Concatenate()([encoder_OUT,encoder_OUT1])
#lambda_function only takes one parameter
distance_layer = layers.Lambda(lambda_function, name="distance_layer")
distance = distance_layer(concat_embeddings)
#distance = tf.fill(dims = (1, 1), value = 1.0)
final_predictions = Concatenate(name='final_predictions')([predict_output,predict_output1])
final_predictions_distance = Concatenate(name='final_predictions_distance')([final_predictions,distance])
#Only training first layer 
###### Huber Removed here RL  
model = Model(inputs=[input_layer], outputs=[final_predictions_distance])
#for idx in range(len(model.layers)):
#  print(model.get_layer(index = idx).name)
#print(model.summary())
# Instantiate an accuracy metric. custom accuracy  type
#accuracyType = tf.keras.metrics.RootMeanSquaredError()

filepath=current_dir +"acc_prob2_vector_distance_boosted/" #modelFile+"_"+"acc"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min') #monitor='val_tpr',
callbacks_list_NN = [checkpoint]

#print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.get_layer('dense').get_weights()))

#First, call compile to configure the optimizer, loss, and metrics to monitor. 1
###### Huber Removed here RL  
model.compile(optimizer=optimizer, loss={'final_predictions_distance': custom_loss},
                     metrics={'final_predictions_distance': [decoder_accuracy_legal,decoder_loss,binaryCrossEntrphy,legal_Network_Pred,custom_loss,mean_sqe_pred,acc_loss,legal_loss,distance_loss]}) #, run_eagerly=True)
print(model.summary())
#print (type(y_train))


'''
num_layers	num_node1	num_node2		test_acc	 legal	    num1_layers	 num1_node1	     num1_node2		test1_acc	  legal1        distance
1	             20     	0	        0.690577626	 1	          1	           20	         0	   	        0.685014188	   1
0.3333333    0.833333333   0.0          0.690577626  1           0.3333333     0.833333333   0.0             0.690577626   1
0.51665676, 0.5047162, 0.49968383,      0.49990097,  0.4997724,  0.51665676,   0.5047162,    0.49968383,    0.49990097,    0.4997724,    0.0
'''


# test_nn = pd.DataFrame({'NoOfAttributes': [6], 'NoOfClasses': [7],'DataSetSize': [42239], 'AttributeType': [0],'EntropyLabel': [0.972495401],
#                 'AvgEntrophyFeatures': [2.6107767],'AvgCorelationBetFeatures': [0.100670496], '2_6_8_TrainingAccuracy': [0.719180584], '2_6_8_TestAccuracy': [0.722301126],
#                 '1_5_0_TrainingAccuracy': [0.685245454], '1_5_0_TestAccuracy': [0.69140625],'2_6_20_TrainingAccuracy': [0.73545754], '2_6_20_TestAccuracy': [0.728693187],'num_layers_N': [0.3333333],
#                 'num_node1_N': [0.833333333],'num_node2_N': [0.0],'NoOfAttributes1': [6], 'NoOfClasses1': [7],'DataSetSize1': [42239], 'AttributeType1': [0],'EntropyLabel1': [0.972495401],
#                 'AvgEntrophyFeatures1': [2.6107767],'AvgCorelationBetFeatures1': [0.100670496], '2_6_8_TrainingAccuracy1': [0.719180584], '2_6_8_TestAccuracy1': [0.722301126],
#                 '1_5_0_TrainingAccuracy1': [0.685245454], '1_5_0_TestAccuracy1': [0.69140625],'2_6_20_TrainingAccuracy1': [0.73545754], '2_6_20_TestAccuracy1': [0.728693187], 'num1_layers_N': [0.3333333],
#                 'num1_node1_N': [0.833333333],'num1_node2_N': [0.0] }, index=['501'])   

# #'test_acc','legal','num_layers_N', 'num_node1_N','num_node2_N','test1_acc','legal1','num1_layers_N', 'num1_node1_N','num1_node2_N'
# t1 = tf.constant([[0.690577626, 1, 0.3333333,0.833333333,0.0,0.685014188, 1, 0.3333333, 0.833333333,0.0]])
# t2 = tf.constant([[0.690577626, 1, 0.3333333,0.833333333,0.0,0.685014188, 1, 0.3333333, 0.833333333,0.0, 0.0]])
# value = model.predict(test_nn)
# f = decoder_loss(t1 , value)
# a = custom_loss(t1 , value)
# b = mean_sqe_pred(t1 , value)
# c = binaryCrossEntrphy(t1 , value)
# d = legal_Network_Pred(t1 , value)
# e = decoder_accuracy_legal(t1 , value)




#model = tf.keras.models.load_model(filepath,custom_objects={'tf': tf}, compile=False)


#model.load_weights(filepath + "weights")

# history = model.fit(X_train, {'final_predictions_distance': y_train[['test_acc','legal','num_layers_N', 'num_node1_N','num_node2_N','test1_acc','legal1','num1_layers_N', 'num1_node1_N','num1_node2_N']]}, validation_split=0.1,  
#                                                                    callbacks=callbacks_list_NN, verbose=2, epochs=1, batch_size=batchSize)
                                                                        
#model.load_weights(filepath + "weights")                            
# history = model.fit(X_train, {'final_predictions_distance': y_train[['test_acc','legal','num_layers_N', 'num_node1_N','num_node2_N','test1_acc','legal1','num1_layers_N', 'num1_node1_N','num1_node2_N']]}, validation_split=0.1,  
#                                                                     callbacks=callbacks_list_NN, verbose=2, epochs=1000, batch_size=batchSize) 
#model.save_weights(filepath + "weights")
model.load_weights(filepath + "weights") 


num_layers,num_node1,num_node2 = randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.getRandomLegalNetwork(divideBy,maxNumOfLayers, minNumOfLayers ) # generate valid normalize network
test_nn = pd.DataFrame({'F1': [0], 'F2': [1],'F3': [0], 'F4': [0],'F5': [1],
                    'F6': [0],'F7': [0], 'F8': [0], 'F9': [0],'F10': [0], 'num_layers_N': [num_layers],
                    'num_node1_N': [num_node1],'num_node2_N': [num_node2]}, index=['50112'])
print("test_nn_fabricated.head() ")
print(test_nn.head())
test_nn_rl = test_nn

print('####################################################################################')
#grid_DataSet = current_dir+"gridpoints_921600_DrivenReward_360.csv"
grid_DataSet = current_dir+"Prob2BaseTestDataSiamese.csv"
#grid_DataSet = current_dir+"Prob2BaseTestDataAccuracyLegal"
#randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.generateEmbeddingsAccuracy(model,grid_DataSet,current_dir)
print('# Test Accuracy For Hundred NN after training#') # Error generate with new formula and normalizaton
#randomNetworkGeneration_UpdatedAccuracyModel.testAccuracyForListOfNN_Mix(model, 500, divideBy, max_accuracy_before_norm, min_accuracy_before_norm, maxNumOfLayers, minNumOfLayers,current_dir)
#randomNetworkGeneration_UpdatedAccuracyModel.testAccuracyForListOfNNLegal(model, 500, divideBy, max_accuracy_before_norm, min_accuracy_before_norm, maxNumOfLayers, minNumOfLayers,24,current_dir )
print('####################################################################################')
#randomNetworkGeneration_UpdatedAccuracyModel.testAccuracyForTrainedListOfNN_Acc(model,testDataToCompareAccuracies,current_dir)
print('####################################################################################')
print('# Train for actor-critic #')
print('####################################################################################')


#agent = actorCriticAgent_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.Agent()
#agent = actorCriticAgent_UpdatedAccuracyModelLamdaGamma01.Agent()

#agent = actorCriticAgent_Gradient_Trained_tau00005OnlyActorCriticFrozen.Agent()
agent = actorCriticAgent_Gradient_Trained_gamma01_tau00005Full.Agent()
#agent.setNoise_std_Vector(NOISE_STD_VECTOR1)
#agent.setNoiseGaussianLayers( vector_noise1,vector_layer1,vector_drop_layer1)


filepath=current_dir +"acc_Siamese/" #modelFile+"_"+"acc"

# #initialize critic_main
# state = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
# actionArray = np.array([0.0,0.0,0.0])
# states = np.array([state])
# actions = np.array([actionArray])
# states = tf.convert_to_tensor(states, dtype= tf.float32)
# actions = tf.convert_to_tensor(actions, dtype= tf.float32)
# agent.critic_main(states, actions)
# agent.critic_main2(states, actions)



# critic_main_weights_file = open(current_dir +"acc_critic_Vector_Siamese_thebe/critic_main_weights.pkl","rb")
# state_h1_w = pickle.load(critic_main_weights_file)
# state_h11_w = pickle.load(critic_main_weights_file)
# state_h111_w = pickle.load(critic_main_weights_file)
# state_h2_w = pickle.load(critic_main_weights_file)
# state_h21_w = pickle.load(critic_main_weights_file)
# state_h211_w = pickle.load(critic_main_weights_file)
# state_h3_w = pickle.load(critic_main_weights_file)
# criticA_output_w = pickle.load(critic_main_weights_file)
# criticV_output_w = pickle.load(critic_main_weights_file)

# critic_main_weights_file.close()

# agent.critic_main.get_layer("state_h1").set_weights(state_h1_w)
# agent.critic_main.get_layer("state_h11").set_weights(state_h11_w)
# agent.critic_main.get_layer("state_h111").set_weights(state_h111_w)
# agent.critic_main.get_layer("state_h2").set_weights(state_h2_w)
# agent.critic_main.get_layer("state_h21").set_weights(state_h21_w)
# agent.critic_main.get_layer("state_h211").set_weights(state_h211_w)
# agent.critic_main.get_layer("state_h3").set_weights(state_h3_w)
# agent.critic_main.get_layer("criticA_output").set_weights(criticA_output_w)
# agent.critic_main.get_layer("criticV_output").set_weights(criticV_output_w)

# agent.critic_main.save_weights(filepath + "test_siamese_FV_weights/" + "critic_main_weights")
# agent.critic_target.load_weights(filepath + "test_siamese_FV_weights/" + "critic_main_weights")



# critic_main_weights_file2 = open(current_dir +"acc_critic_Vector_Siamese_thebe/critic_main2_weights.pkl","rb")
# state_h1_w2 = pickle.load(critic_main_weights_file2)
# state_h11_w2 = pickle.load(critic_main_weights_file2)
# state_h111_w2 = pickle.load(critic_main_weights_file2)
# state_h2_w2 = pickle.load(critic_main_weights_file2)
# state_h21_w2 = pickle.load(critic_main_weights_file2)
# state_h211_w2 = pickle.load(critic_main_weights_file2)
# state_h3_w2 = pickle.load(critic_main_weights_file2)
# criticA_output_w2 = pickle.load(critic_main_weights_file2)
# criticV_output_w2 = pickle.load(critic_main_weights_file2)

# critic_main_weights_file2.close()

# agent.critic_main2.get_layer("state_h1").set_weights(state_h1_w2)
# agent.critic_main2.get_layer("state_h11").set_weights(state_h11_w2)
# agent.critic_main2.get_layer("state_h111").set_weights(state_h111_w2)
# agent.critic_main2.get_layer("state_h2").set_weights(state_h2_w2)
# agent.critic_main2.get_layer("state_h21").set_weights(state_h21_w2)
# agent.critic_main2.get_layer("state_h211").set_weights(state_h211_w2)
# agent.critic_main2.get_layer("state_h3").set_weights(state_h3_w2)
# agent.critic_main2.get_layer("criticA_output").set_weights(criticA_output_w2)
# agent.critic_main2.get_layer("criticV_output").set_weights(criticV_output_w2)

# agent.critic_main2.save_weights(filepath + "test_siamese_FV_weights/" + "critic_main2_weights")
# agent.critic_target2.load_weights(filepath + "test_siamese_FV_weights/" + "critic_main2_weights")

 
# filepath=current_dir +"acc/"
# agent.actor_main.load_weights(filepath + "test_siamese_weights/" + "actor_main_weights")
# agent.actor_target.load_weights(filepath + "test_siamese_weights/" + "actor_target_weights")
# agent.critic_main.load_weights(filepath + "test_siamese_weights/" + "critic_main_weights")
# agent.critic_main2.load_weights(filepath + "test_siamese_weights/" + "critic_main2_weights")
# agent.critic_target.load_weights(filepath + "test_siamese_weights/" + "critic_target_weights")
# agent.critic_target2.load_weights(filepath + "test_siamese_weights/" + "critic_target2_weights")

#grid_DataSet = current_dir+"list_of_gridData_360_Siamese_325_RewardV.csv"
#randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.generateEmbeddingsActor360(grid_DataSet, agent,current_dir,accuracy_model)
#randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.generateEmbeddingsReward(grid_DataSet, agent,accuracy_model, current_dir)
#grid_DataSet = current_dir+"gridpoints_Siameese_ActionReward.csv"
#randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.generateEmbeddingsCriticTestV(grid_DataSet, agent,current_dir)
#agent = reinforcementTraining_Gradient_Trained_tau00005CriticFrozen.rlTraining(test_nn_rl, accuracy_model, agent, divideBy, maxNumOfLayers, minNumOfLayers,current_dir)


filepath=current_dir +"acc_Siamese/" #modelFile+"_"+"acc"         
agent.actor_main.load_weights(filepath + "test_siamese_FV_weights/" + "actor_main_weights")
agent.actor_target.load_weights(filepath + "test_siamese_FV_weights/" + "actor_target_weights")
agent.critic_main.load_weights(filepath + "test_siamese_FV_weights/" + "critic_main_weights")
agent.critic_main2.load_weights(filepath + "test_siamese_FV_weights/" + "critic_main2_weights")
agent.critic_target.load_weights(filepath + "test_siamese_FV_weights/" + "critic_target_weights")
agent.critic_target2.load_weights(filepath + "test_siamese_FV_weights/" + "critic_target2_weights")
#agent = reinforcementTraining_Gradient_Trained_tau00005CriticFrozen.rlTraining(test_nn_rl, model, agent, divideBy, maxNumOfLayers, minNumOfLayers,current_dir)
#randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.generateEmbeddingsReward(grid_DataSet, agent,model, current_dir)


agent = reinforcementTraining_Gradient_Trained_tau00005CriticFrozen.rlTraining(test_nn_rl, accuracy_model, feature_Conversion_model, agent, divideBy, maxNumOfLayers, minNumOfLayers,current_dir)

print('####################################################################################')
print('# Training FINISHED for actor-critic #')
print('####################################################################################')
############################################################################
## Testing the Policy
# This is for testing after tarining is done.
# test for network 2,10,10 expecting accuracy 1
# Which is 0.08, 0.4, 0.4 and accuracy = 1
total_reward = 0
#-----------------
# If you want to directly load the agent weights the uncomment followings
        # Save weights in a file 

'''
filepath=current_dir +"acc_Siamese/" #modelFile+"_"+"acc"         
agent.actor_main.load_weights(filepath + "test_siamese_FV_weights/" + "actor_main_weights")
agent.actor_target.load_weights(filepath + "test_siamese_FV_weights/" + "actor_target_weights")
agent.critic_main.load_weights(filepath + "test_siamese_FV_weights/" + "critic_main_weights")
agent.critic_main2.load_weights(filepath + "test_siamese_FV_weights/" + "critic_main2_weights")
agent.critic_target.load_weights(filepath + "test_siamese_FV_weights/" + "critic_target_weights")
agent.critic_target2.load_weights(filepath + "test_siamese_FV_weights/" + "critic_target2_weights")

#grid_DataSet = current_dir+"9NN_FeatureVectorTraining.csv"
grid_DataSet = current_dir+"gridpoints_Siameese_FV8_Actor.csv"
#randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.generateEmbeddingsActorFromVector(grid_DataSet, agent,feature_Conversion_model,current_dir,accuracy_model)
grid_DataSet = current_dir+"gridpoints_Siameese_FV9_Action.csv"
randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.generateEmbeddingsCriticVOnGrid(grid_DataSet, agent,feature_Conversion_model, current_dir,accuracy_model)
#randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.generateEmbeddingsCriticTest(grid_DataSet, agent,current_dir)
#randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.generateEmbeddingsCritic_(grid_DataSet, agent,current_dir)
#grid_DataSet = current_dir+"test_DataSet10.csv"


#randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.generateEmbeddingsCriticCritic_main2_(grid_DataSet, agent,current_dir)
#randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.generateEmbeddingsCriticCritic_target_(grid_DataSet, agent,current_dir)
#randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.generateEmbeddingsCriticCritic_target2_(grid_DataSet, agent,current_dir)

print ("=====================   Generated All Critic  using gradienttape training")
# experiment start #################################################################################

print('####################################################################################')
print('#  Critic  generated for Embeddings #')
print('####################################################################################')

#-------------------
random100 = True  # Use random NN of 100 NN to do policy evaluation
writePolicyOutPutFile = current_dir+"38234EpisodsRandomRLRandomPolicyRandom100NN/policyOutPutFor.csv"
myPolicyOutPutFile = open(writePolicyOutPutFile, 'w')
##  TODO - CHange the headings as per the results
headerPolicyOutPutFile = ['encoded_x','encoded_y', 'layers', 'nnode1','nnode2','accuracy','reward','total_reward','critic_value_state','critic_value_state_action','actor_value_1','actor_value_2','actor_value_3'] 
writerPolicyOutPut = csv.DictWriter(myPolicyOutPutFile, fieldnames=headerPolicyOutPutFile) 
writerPolicyOutPut.writeheader()
file_utility_values_R = open(current_dir+'rl_results/utility_values_R_1_1.csv', 'a')
if random100:
    #=========
    for index1 in range(100):
        sameStartNode = False
        encoded_x, encoded_y,n_layers,num_node1,num_node2, prev_accuracy,reward,total_reward,qValue1, qValue2, qActions_1, qActions_2, qActions_3= helperFunctions_CriticFrozen.getPolicyoutput(index1,model,accuracy_model,agent,divideBy,maxNumOfLayers, minNumOfLayers, decoder_model,current_dir,file_utility_values_R)
        # save into a csv file
        writerPolicyOutPut.writerow({ 'encoded_x' : encoded_x, 
                'encoded_y' : encoded_y, 
                'layers' : n_layers,
                'nnode1' : num_node1,
                'nnode2' : num_node2,
                'accuracy' : prev_accuracy, 
                'reward': reward,
                'total_reward': total_reward,
                'critic_value_state' : qValue1,
                'critic_value_state_action' : qValue2,
                'actor_value_1' : qActions_1,
                'actor_value_2' : qActions_2,
                'actor_value_3' : qActions_3
                })
        state, num_layers,num_node1,num_node2, accuracy_state = randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.getStateAndAccuracy(model,divideBy,maxNumOfLayers, minNumOfLayers)
else:
    #  same as testNN
    sameStartNode = True                     
    encoded_x, encoded_y,n_layers,num_node1,num_node2, prev_accuracy,reward,total_reward,qValue1,qValue2,qActions_1, qActions_2,qActions_3 = helperFunctions_CriticFrozen.getPolicyoutput(0,model,accuracy_model,agent,divideBy,maxNumOfLayers, minNumOfLayers, decoder_model,current_dir,file_utility_values_R )
    # save into a csv file
    writerPolicyOutPut.writerow({ 'encoded_x' : encoded_x, 
            'encoded_y' : encoded_y, 
            'layers' : n_layers,
            'nnode1' : num_node1,
            'nnode2' : num_node2,
            'accuracy' : prev_accuracy, 
            'reward': reward,
            'total_reward': total_reward,
            'critic_value' : qValue1,
            'critic_value_state_action' : qValue2,
            'actor_value_1' : qActions_1,
            'actor_value_2' : qActions_2,
            'actor_value_3' : qActions_3
            })
myPolicyOutPutFile.close()
file_utility_values_R.close()
'''