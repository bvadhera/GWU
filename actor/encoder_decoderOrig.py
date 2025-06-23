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
      node = "node_" + str(i+1)
      attrib = df_max[node]
      maxNumOfNodes.append(int(attrib.max()))
    print (max(maxNumOfNodes))
    #get max number of nodes
    return max(maxNumOfNodes)

def removeColumns(properties):
      properties.remove('training_loss')
      properties.remove('training_accuracy')
      properties.remove('val_loss')
      properties.remove('val_accuracy')
      properties.remove('test_loss')
      properties.remove('test_acc')
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

def splitAndUpdateNodes(mydf):
      maxLayers = mydf['num_layers'].max()
      print("Maximum value in column 'maxLayers': " )
      print(maxLayers)
      # iterating over rows using iterrows() function
      for index, row in mydf.iterrows():
        #print('num_layers')
        noOfLayers = row.loc['num_layers']
        #print(noOfLayers)
        noOfnodes = row.loc['num_nodes']
        a = noOfnodes.replace("[", "")
        b = a.replace("]", "")
        c = b.replace(" ", "")
        nodes_list =  c.split(",")
        #print("noOfnodes") 
        #print(nodes_list)

         # using map and int
        noOfnodes = list(map(int, nodes_list))
        #print("AGAIN noOfnodes as int")
        #print ("-----")
        #print(noOfnodes)
        #print ("-----")
        # iterate as many layers and create node columns 
        i = 0
        for i in range(maxLayers):
          if i == 0:
            node = "node_" + str(i+1)
            #print(noOfnodes[i])
            mydf.loc[index,node] = int(noOfnodes[i])
          else :
            node = "node_" + str(i+1)
            if 0 <= i < len(noOfnodes):
              #print (len(noOfnodes))
              mydf.loc[index,node] = int(noOfnodes[i])
            else:
              mydf.loc[index,node] = int(0)     
      return mydf


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
inputNN_Architectures_DataSet = current_dir+"inputData/inputNN_Architectures_DataSet.csv"
print (inputNN_Architectures_DataSet)
# Open a csv reader called DictReader
# iterate over each line as a ordered dictionary and print only few column by column name
with open(inputNN_Architectures_DataSet, 'r') as read_obj:
    csv_dict_reader = DictReader(read_obj)
    ## Outer Loop starts
    # For each dataset(row) generate 50 random NN such as:
    # For Each dataset generate autoencoder
    # Save 50 NN arcitecture (embedding points into a new file)
    #Save into csv file 
    for row in csv_dict_reader:
        filename = row['Filename']
        f1 = row['F1']
        f2 = row['F2']
        f3 = row['F3']
        f4 = row['F4']
        f5 = row['F5']
        f6 = row['F6']
        f7 = row['F7']
        f8 = row['F8']
        f9 = row['F9']
        f10 = row['F10']
        # Hardcoded as we need to normalize file 'nn_architectures_SomervilleHappinessSurvey2015'to spreed nodes values based upon max num of layers 
        #filename = "nn_architectures_SomervilleHappinessSurvey2015.csv"
        print(filename)
        #For each Dataset File creat test and train data
        current_file = current_dir + "NN_Architectures/" + filename
        df = pd.read_csv(current_file)
    
        print(df.head())
        properties = list(df.columns.values)
        print(properties)
        # remove properties 'training_loss','training_accuracy',  'val_loss',
        #  'val_accuracy', 'test_loss' , 'test_acc', 'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'
        updatedProps = removeColumns(properties)
        print(updatedProps)
        mydf = df[updatedProps]
        #print("mydf")
        #print(mydf)
        #Normalize the data
        
        #print(mydf)
        #drop
        mydf = splitAndUpdateNodes(mydf)
        newProperties = list(mydf.columns.values)
        #print(newProperties)
        # remove properties 'training_loss','training_accuracy',  'val_loss', 'val_accuracy', 'test_loss' , 'test_acc'
        newProperties.remove('num_nodes')

        #print("newProperties")
        #print(newProperties)
        train_test_df = mydf[newProperties]
        print(train_test_df)
        # Q - why not integer value in the mydef
         
        # Keeping both train and test as same as we need embedding points (x,y) in 2D space
         # Q - should train and test both same
        x_train = train_test_df[newProperties].to_numpy()
        #normalize the data
        devideBy = getMaxNumOfNodesInDataSet(train_test_df)
        x_train = x_train/devideBy
        x_test =  x_train

        print("train_test_df ============================")
        print (x_train[:10])
        print (x_test[:10])
        print (x_train.shape)
        print (x_test.shape)
        #NN embedding File Name 
        writeFile = current_dir + "embeddings/emb_"
        writeFile += filename
        print("writeFile")
        print(writeFile)
        myFile = open(writeFile, 'w')
        with myFile:    
            header = ['x','y', 'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10']
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
            print (history.history)
            encoded_data = autoencoder.encoder(x_test)
            print (type(encoded_data))
            myXYvalues = encoded_data.numpy()
            print("encoded_data=======================================>")
            print(myXYvalues)
            # save into a csv file
            i = 0
            j = 0
            for i in range(0, 50): 
              x = myXYvalues.item(i,0)
              y = myXYvalues.item(i,1)
              writer.writerow({'x' : x,  
                            'y': y, 
                            'F1':f1,'F2':f2,'F3':f3,'F4': f4,'F5':f5,
                            'F6':f6,'F7':f7,'F8':f8,'F9':f9,'F10':f10})
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
            fileToDump = "recovered_" + filename
            print (fileToDump)
            xy_df.to_csv(current_dir + "recovered/"+fileToDump)
 
        myFile.close()
