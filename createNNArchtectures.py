import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import csv
import random
from csv import DictReader
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder



current_dir  = "/home/bvadhera/huber/"
inputTrainDataDetFile = current_dir+"inputData/inputTrainDataDetFile.csv"
print (inputTrainDataDetFile)

def normalize(df_,features):
    #result = df.copy()
    for feature_name in features:
        max_value = df_[feature_name].max()
        min_value = df_[feature_name].min() ;print("min =",min_value,"\t max=",max_value)
        df_[feature_name+"_N"] = (df[feature_name] - min_value) / (max_value - min_value)

#Max attrib from this dataset File
# load The TrainingDataSet File
def getMaxNodesAmongAllDataSets():
    df_max = pd.read_csv(inputTrainDataDetFile)
    properties = list(df_max.columns.values)
    print(properties)
    attrib = df_max['NoOfAttributes']
    return attrib.max()
    #return int(10)  # as max attributes are 10

maxNodesGlobal = getMaxNodesAmongAllDataSets()
print("MaxNodesAmongAllDataSets", maxNodesGlobal)


# Open a csv reader called DictReader
# iterate over each line as a ordered dictionary and print only few column by column name
with open(inputTrainDataDetFile, 'r') as read_obj:
    csv_dict_reader = DictReader(read_obj)
    ## Outer Loop starts
    # For each dataset(row) generate 50 random NN such as:
    #     For Each dataset generate Fully connected Sequential NN with  random num of layers and num of nodes
    #        make model and train and get accuracy 
    #        store/save Filename/No of attributes / No of classes / Trained data points / Accuracy / 
    #               save 50 NN arcitecture (Seqential, Dense, Num of Layers, Num Nodes on Each Layer, Accuracy)
    #Save into csv file 
    for row in csv_dict_reader:
        filename = row['Filename']
        noOfAttributes = row['NoOfAttributes']
        noOfClasses = row['NoOfClasses']
        
        dataPoints = row['DataPoints']
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

        print('####################################################################################')
        print("filename, noOfAttributes, noOfClasses, dataPoints,f1 ,f2 ,f3 ,f4 ,f5 ,f6 ,f7 ,f8 ,f9 ,f10")
        print(filename, noOfAttributes, noOfClasses, dataPoints,f1 ,f2 ,f3 ,f4 ,f5 ,f6 ,f7 ,f8 ,f9 ,f10)
        print('####################################################################################')
        #For each Dataset File create test and train data
        df = pd.read_csv(current_dir + "inputData/" + filename)
        print(df.head())
        old_properties = list(df.columns.values)
        print(old_properties)
        old_properties.remove('Class')
        print(old_properties)
        X_beforeNormalization = df[old_properties]

        #normalize dataSet properties before training
        normalize(X_beforeNormalization,old_properties)
        old_new_properties = list(X_beforeNormalization.columns.values)
        for element in old_properties:
          if element in old_new_properties:
            old_new_properties.remove(element)
        print(old_new_properties)
        normalizedProperties = old_new_properties
        print('####################################################################################')
        print("normalizedProperties")
        print(normalizedProperties)
        print('####################################################################################')
        X = X_beforeNormalization[normalizedProperties]
        y = df['Class']
        print("X.head()")
        print(X.head())
        print("y.head()")
        print(y.head())

        #Find the min value if that value is not 0 then substract each element with 1 so that label starts from 0 onwards
        max_ClassValue = y.max()
        min_ClassValue = y.min()
        print("min_ClassValue =",min_ClassValue,"\t max_ClassValue=",max_ClassValue)
        if (int(noOfClasses)  == max_ClassValue) : 
          if (min_ClassValue != 0) :
            # reduce all class values by 1
            for i in y.index:
              df.at[i, "Class"] -= 1
        print(y)
        # Divide the data into test and train sets
        a = random.randint(1,50)
        print ("random_state")
        print (a)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=a)
        print("The type of x_train is : ",type(X_train))
        # Get Batch size from X_train
        bSize = len(X_train)
        print("The type of bSize is : ",type(bSize), bSize)
        batchSize = 0
        if (bSize > 5000 ):
          batchSize = 5000
        elif (bSize > 500 ):
          batchSize = 500
        elif (bSize > 256 ):
          batchSize = 256
        elif (bSize > 128 ):
          batchSize = 128
        elif (bSize > 64 ):
          batchSize = 64
        elif (bSize > 32 ):
          batchSize = 32 
        else :
          batchSize = bSize
        print ("batchSize")
        print (batchSize)
        #NN Architecture File Name 
        writeFile = current_dir + "outputData"+"/nn_architectures_"
        writeFile += filename
        print("writeFile")
        print(writeFile)
        myFile = open(writeFile, 'w')
        with myFile:    
            header = ['num_layers', 'num_nodes', 'training_loss','training_accuracy',  'val_loss', 'val_accuracy', 'test_loss', 
                      'test_acc','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10']
            writer = csv.DictWriter(myFile, fieldnames=header)    
            writer.writeheader()
            #For Loop for 50 NN
            n = 50 
            for i in range(50):
              print (str(i) + " out of 50 ")
              # Create a NN with one input layer, 1 or 2 dense layers and 1 output layer
              n_layers = np.random.randint(1,3,1)[0]
              min = 1
              # upper limit will be max of NoOfAttributes among all the datasets
              max = 3 * int(maxNodesGlobal)
              layers = []
              nodes = []
              for l in range(n_layers):
                n_nodes = np.random.randint(min,max,1)[0]
                nodes.append(n_nodes)
                layers.append(Dense(int(n_nodes), activation=tf.nn.leaky_relu))
                
              layers.append(Dense(int(noOfClasses), activation='softmax'))
              # Instantiate a simple classification model
              model = tf.keras.Sequential(layers)
              # Instantiate a logistic loss function that expects integer targets.
              lossfuntion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
              # Instantiate an accuracy metric.
              accuracyType = tf.keras.metrics.SparseCategoricalAccuracy()
              # Instantiate an optimizer.
              learning_rate = 0.001
              optimizer = tf.keras.optimizers.Adam(learning_rate)
              filepath=current_dir+"outputOfRun/" +"NN." #modelFile+"_"+"NN."
              checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') #monitor='val_tpr',
              callbacks_list_NN = [checkpoint]
              #First, call compile to configure the optimizer, loss, and metrics to monitor.
              model.compile(optimizer=optimizer, loss=lossfuntion, metrics=[accuracyType])
              history = model.fit(X_train, y_train, validation_split=0.1,  callbacks=callbacks_list_NN, verbose=2, epochs=1000, batch_size=batchSize)
              #history = model.fit(X_train, y_train, validation_split=0.1, verbose=2, epochs=1000, batch_size=batchSize)
              training_loss = history.history["loss"]
              training_accuracy = history.history["sparse_categorical_accuracy"]
              val_loss = history.history["val_loss"]
              val_accuracy = history.history["val_sparse_categorical_accuracy"]
              test_loss, test_acc = model.evaluate(X_test, y_test)
              print('####################################################################################')
              print('Test accuracy:', test_acc)
              print('####################################################################################')
              # save into a csv file
              writer.writerow({'num_layers' : n_layers, 'num_nodes': nodes, 
                               'training_loss' : training_loss[-1], 
                               'training_accuracy' : training_accuracy[-1], 
                               'val_loss' : val_loss[-1], 
                               'val_accuracy' : val_accuracy[-1], 
                               'test_loss': test_loss,
                               'test_acc' : test_acc,'F1':f1,'F2':f2,'F3':f3,'F4': f4,'F5':f5,
                               'F6':f6,'F7':f7,'F8':f8,'F9':f9,'F10':f10})
        myFile.close()