import tensorflow as tf
import numpy as np
import math
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






def normal_round(n):
    if n - math.floor(n) < 0.5:
        return math.floor(n)
    return math.ceil(n)


def normalize(df_,features):
    #result = df.copy()
    for feature_name in features:
        max_value = df_[feature_name].max()
        min_value = df_[feature_name].min() ;print("min =",min_value,"\t max=",max_value)
        df_[feature_name+"_N"] = (df[feature_name] - min_value) / (max_value - min_value)


current_dir  = "/home/bvadhera/huber/"
#Access to all 50 NN created from Gradients
inputTrainDataGradNNFile = current_dir+"outputData/nn_grad_architectures_0000100000.csv"
print (inputTrainDataGradNNFile)

#NN Architecture File Name where new NN with Gradients will be saved with new found accuracies
writeFile = current_dir + "outputData"+"/nn_Grad_Acc_"
writeFile += "BasketballDataset.csv"
print("writeFile")
print(writeFile)
myFile = open(writeFile, 'w')
header = ['num_layers', 'num_nodes', 'training_loss','training_accuracy',  'val_loss', 'val_accuracy', 'test_loss' , 
          'test_acc','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10']
writer = csv.DictWriter(myFile, fieldnames=header)    
writer.writeheader() 

# Open a csv reader called DictReader to read Gradient NN created
# iterate over each line as a ordered dictionary and print only few column by column name
with open(inputTrainDataGradNNFile, 'r') as read_obj:
    csv_dict_reader = DictReader(read_obj)
    ## Outer Loop starts
    # For each dataset(row) generate 50 random NN such as:
    #     For Each dataset generate Fully connected Sequential NN with  random num of layers and num of nodes
    #        make model and train and get accuracy 
    #        store/save Filename/No of attributes / No of classes / Trained data points / Accuracy / 
    #               save 50 NN arcitecture (Seqential, Dense, Num of Layers, Num Nodes on Each Layer, Accuracy)
    #Save into csv file 
    # Chec with Dr Huber
    devideBy = int(25)
    for row in csv_dict_reader:
      # multiply by devideBy factor as it was used before and round it        
        num_layers = normal_round((float((row['num_layers'])))* devideBy)
        num_node1 = normal_round((float((row['num_node1'])))* devideBy)
        num_node2 = normal_round((float((row['num_node2'])))* devideBy)
        
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
        print("num_layers, num_node1, num_node2, f1 ,f2 ,f3 ,f4 ,f5 ,f6 ,f7 ,f8 ,f9 ,f10")
        print(num_layers, num_node1, num_node2 ,f1 ,f2 ,f3 ,f4 ,f5 ,f6 ,f7 ,f8 ,f9 ,f10)
        print('####################################################################################')
        #For each NN   create test and train data  from oneHotVector file BasketballDataset.csv
        df = pd.read_csv(current_dir + "inputData/BasketballDataset.csv")
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
        # noOfClasses in BasketballDataset
        noOfClasses = 5
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
           
        layers = []
        nodes = []
        nodes.append(num_node1)
        nodes.append(num_node2)
        layers.append(Dense(int(num_node1), activation=tf.nn.leaky_relu))
        layers.append(Dense(int(num_node2), activation=tf.nn.leaky_relu))
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
        writer.writerow({'num_layers' : num_layers, 'num_nodes': nodes, 
                          'training_loss' : training_loss[-1], 
                          'training_accuracy' : training_accuracy[-1], 
                          'val_loss' : val_loss[-1], 
                          'val_accuracy' : val_accuracy[-1], 
                          'test_loss': test_loss,
                          'test_acc' : test_acc,'F1':f1,'F2':f2,'F3':f3,'F4': f4,'F5':f5,
                          'F6':f6,'F7':f7,'F8':f8,'F9':f9,'F10':f10})
myFile.close()
inputTrainDataGradNNFile.close()