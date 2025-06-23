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
import math
import keras.backend as K


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


def normal_round(n):
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)

def getMaxNumOfNodesInDataSet(df_max):
    properties = list(df_max.columns.values)
    print(properties)
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


def custom_loss(y_true, y_pred):
    print("type(y_true)")
    print(type(y_true) )
    print("type(y_pred)")
    print(type(y_pred) )
    # convert y_true to tensorflow.python.keras.engine.keras_tensor.KerasTensor
    print ('y_true.shape:', y_true.shape)  # shape 
    print ('y_pred.shape:',y_pred.shape)   # shape 
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    #def loss(y_true,y_pred):
    weight = 0.0
    loss =  K.mean(K.square( y_true[:,:3] - y_pred[:,:3]), axis=-1)
    # Return a function
    return loss 

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      #Hidden Layer
      layers.Dense(latent_dim, activation=tf.nn.leaky_relu),
      layers.Dense(latent_dim, activation=tf.nn.leaky_relu),
      #output Layer
      layers.Dense(int(2), activation=tf.nn.leaky_relu)
    ])
    
    #In parellel Step 1 and Step 2

    # Step1 
    # We will get x,y value here
    # also goes to onehotencoding (To original NN) and trained with the original performance
    #
    #Step2
    self.decoder = tf.keras.Sequential([
      layers.Dense(latent_dim, activation=tf.nn.leaky_relu),
      layers.Dense(latent_dim, activation=tf.nn.leaky_relu),
      #output Layer
      layers.Dense(int(3), activation=tf.nn.leaky_relu)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

# Dense Layers nodes
latent_dim = 10
autoencoder = Autoencoder(latent_dim) 

current_dir  = "/home/bvadhera/huber/"
current_file = ""
# load The TrainingDataSet File
inputNN_Architectures_DataSet = current_dir+"combined_nn_architectures_OneHot.csv"
print (inputNN_Architectures_DataSet)
# Open a csv reader called DictReader
# iterate over each line as a ordered dictionary and print only few column by column name

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

train_test_df = df[normalizeProp]
print(train_test_df.head())
# Q - why not integer value in the mydef
  
x_train = train_test_df[normalizeProp].to_numpy()

#normalize the data
devideBy = getMaxNumOfNodesInDataSet(train_test_df) + 2
x_train = x_train/devideBy
x_test =  x_train

print("train_test_df ============================")
print (x_train[:10])
print (x_test[:10])
print (x_train.shape)
print (x_test.shape)
print (type(x_train))
print (type(x_test))

writeFile = current_dir + "results_" + "combined_nn_architectures_OneHot.csv"
myFile = open(writeFile, 'w')
with myFile:    
    header = ['x','y']
    writer = csv.DictWriter(myFile, fieldnames=header)    
    writer.writeheader()
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    #autoencoder.compile(optimizer, loss=losses.MeanSquaredError())
    autoencoder.compile(optimizer, loss=custom_loss)
    history = autoencoder.fit(x_train, x_train,
        epochs=1000,
        batch_size=50,
        shuffle=True,
        validation_data=(x_test, x_test))
    print ("history.history")
    print (history.history)
    encoded_data = autoencoder.encoder(x_test)
    print (type(encoded_data))
    myXYvalues = encoded_data.numpy()
    print("encoded_data=======================================>")
    print(myXYvalues)
    # save into a csv file
    i = 0
    j = 0
    for i in range(0, 500): 
      x = myXYvalues.item(i,0)
      y = myXYvalues.item(i,1)
      writer.writerow({'x' : x,  
                    'y': y})
    decoded_data = autoencoder.decoder(encoded_data).numpy()
    #recoverBack
    print("decoded_data=======================================>")
    print (devideBy)
    print(decoded_data*devideBy)
    print (type(decoded_data))
    # convert array into dataframe
    scale_df = pd.DataFrame(decoded_data*devideBy)
    nn_df = np.vectorize(normal_round)(scale_df)
    print (type(nn_df))
    print ("nn_df============================================>")
    print (nn_df)
    xy_df = pd.DataFrame(data=nn_df)
    print ("xy_df============================================>")
    print (xy_df)
    # save the dataframe as a csv file
    fileToDump = "recovered_" + "combined_nn_architectures_OneHot"
    print (fileToDump)
    xy_df.to_csv(current_dir + "recovered/"+fileToDump)

myFile.close()
