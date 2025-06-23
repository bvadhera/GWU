import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
print(keras.__version__)
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
    weight = 0.0
    #loss =  K.mean(K.square( y_true[:,:3] - y_pred[:,:3]), axis=-1)  + weight*(K.square(y_true[:,3:] - y_pred[:,3:]))
    loss =  K.mean(K.square( y_true[:,:3] - y_pred[:,:3]), axis=-1)
    # Return a function
    return loss

# Accuracy Matrics First
def decoder_accuracy(y_true,y_pred):
    #predx = K.batch_get_value(y_pred[:,:3])  # tensor for num_layers, node1 & node2 after decoder
    #truex = K.batch_get_value(y_true[:,:3]) # tensor for num_layers, node1 & node2 before decoder
    denorm_y_true = y_true[:,:3] * 25 
    denorm_y_pred = y_pred[:,:3] * 25 

    diff = K.equal(K.mean(K.round(denorm_y_true - denorm_y_pred),  axis=-1), 0)
    return(K.cast(diff,tf.float32))


# Accuracy Matrics Second
def mean_sqe_pred(y_true,y_pred): 
    return K.square(y_true[:,3:] - y_pred[:,3:])

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


# Dense Layers nodes for encoder and decoder
latent_dim = 10
current_dir  = "/home/bvadhera/huber/"
# load The TrainingDataSet File
inputNN_Architectures_DataSet = current_dir+"combined_nn_architectures_OneHot.csv"
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
trueProperties.remove('test_acc')
print("trueProperties")
print(trueProperties)
y = df[trueProperties]
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



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=a)
print("X_train.head() ")
print(X_train.head())
print("X_test.head() ")
#print(X_test.head())
print("y_train.head() ")
print(y_train.head())
#print("y_test.head() ")
#print(y_test.head())

print("type(y_train)")
print(type(y_train) )

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
#predict_layer = Concatenate()([embedding_layer,split[0]])
#print ('predict_layer.shape:', predict_layer.shape)
#Hidden Layers
#hiddenP_layer_1 = Dense(int(10), activation=tf.nn.leaky_relu)(predict_layer)
#print ('hiddenP_layer_1.shape:', hiddenP_layer_1.shape)
#hiddenP_layer_2 = Dense(int(5), activation=tf.nn.leaky_relu)(hiddenP_layer_1)
#print ('hiddenP_layer_2.shape:', hiddenP_layer_2.shape)
#Call Model to get predicted Accuracy as output
#accuracy_layer = Dense(int(1), activation='sigmoid')(predict_layer)
#print ('predict_outpur_layer.shape:', accuracy_layer.shape)
#+++++++++++++++++++++++++++++++++++++
# Building Final  JOINT TRACK  
#++++++++++++++++++++++++++++++++
#Merge predict_layer (predicted Accuracy) back with decoded_layer
#predict_output = Concatenate()([decoded_layer, accuracy_layer])
#print ('accuracy_layer.shape:', accuracy_layer.shape)


#Call Model to get Final Accuracy as output with custom MSE loss function
##Defining the model by specifying the input and output layers
model = Model(inputs=input_layer, outputs=decoded_layer)



# Instantiate an accuracy metric. custom accuracy  type
#accuracyType = tf.keras.metrics.RootMeanSquaredError()
# Instantiate an optimizer.
learning_rate = 0.01
optimizer = tf.keras.optimizers.Adam(learning_rate)
filepath=current_dir +"acc." #modelFile+"_"+"acc."
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max') #monitor='val_tpr',
callbacks_list_NN = [checkpoint]
#First, call compile to configure the optimizer, loss, and metrics to monitor. 
model.compile(optimizer=optimizer, loss=custom_loss, metrics=[decoder_accuracy , mean_sqe_pred])
history = model.fit(X_train, y_train, validation_split=0.1,  callbacks=callbacks_list_NN, verbose=2, epochs=1000, batch_size=batchSize)
print (history.history)
training_loss = history.history["loss"]
#custom_loss  = history.history["custom_loss"]
val_loss = history.history["val_loss"]
decoder_accuracy = history.history["decoder_accuracy"]
mean_sqe_pred = history.history["mean_sqe_pred"]
val_decoder_accuracy = history.history["val_decoder_accuracy"]
val_mean_sqe_pred = history.history["val_mean_sqe_pred"] 
#test_loss, test_decoder_accuracy, test_mean_sqe_pred = model.evaluate(X_test, y_test)
print('####################################################################################')
#print('Test accuracy:', test_decoder_accuracy, test_mean_sqe_pred)
print('####################################################################################')
# save into a csv file
writer.writerow({ 'training_loss' : training_loss[-1], 
                  'val_loss' : val_loss[-1], 
                  'decoder_accuracy' : decoder_accuracy[-1],
                  'mean_sqe_pred' : mean_sqe_pred[-1],
                  'val_decoder_accuracy' : val_decoder_accuracy[-1],
                  'val_mean_sqe_pred' : val_mean_sqe_pred[-1], 
                  #'test_loss': test_loss,
                  #'test_decoder_accuracy' : test_decoder_accuracy,
                  #'test_mean_sqe_pred' : test_mean_sqe_pred
                  })
myFile.close()