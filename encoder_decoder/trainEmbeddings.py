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
inputTrainDataDetFile = current_dir+"embeddings/inputEmbeddings.csv"
print (inputTrainDataDetFile)

def normalize(df_,features):
    #result = df.copy()
    for feature_name in features:
        max_value = df_[feature_name].max()
        min_value = df_[feature_name].min() ;print("min =",min_value,"\t max=",max_value)
        df_[feature_name+"_N"] = (df[feature_name] - min_value) / (max_value - min_value)



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
        #noOfAttributes = row['NoOfAttributes']
        #noOfClasses = row['NoOfClasses']
        #dataPoints = row['DataPoints']
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
        print("filename, f1 ,f2 ,f3 ,f4 ,f5 ,f6 ,f7 ,f8 ,f9 ,f10")
        print(filename, f1 ,f2 ,f3 ,f4 ,f5 ,f6 ,f7 ,f8 ,f9 ,f10)
        print('####################################################################################')
        #For each Dataset File create test and train data
        df = pd.read_csv(current_dir + "embeddings/" + filename)
        print(df.head())
        old_properties = list(df.columns.values)
        print(old_properties)
        old_properties.remove('F1')
        old_properties.remove('F2')
        old_properties.remove('F3')
        old_properties.remove('F4')
        old_properties.remove('F5')
        old_properties.remove('F6')
        old_properties.remove('F7')
        old_properties.remove('F8')
        old_properties.remove('F9')
        old_properties.remove('F10')

        print(old_properties)
        X_beforeNormalization = df[old_properties]
        '''
        #normalize dataSet properties before training
        normalize(X_beforeNormalization,old_properties)
        old_new_properties = list(X_beforeNormalization.columns.values)
        
        for element in old_properties:
          if element in old_new_properties:
            old_new_properties.remove(element)
        print(old_new_properties)
        normalizedProperties = old_new_properties
        '''
        y_properties = list(old_properties)
        
        old_properties.remove('test_acc')

        y_properties.remove('x')
        y_properties.remove('y')

        print('####################################################################################')
        print ("x_properties")
        print (old_properties)
        print ("y_properties")
        print (y_properties)
        print('####################################################################################')
    
        X = X_beforeNormalization[old_properties]
        y = X_beforeNormalization[y_properties]
        print("X.head()")
        print(X.head())
        print("y.head()")
        print(y.head())
        print ()
        # Divide the data into test and train sets
        a = random.randint(1,50)
        print ("random_state")
        print (a)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=a)
        print("The type of x_train is : ",type(X_train))
        # Get Batch size for X_train
        batchSize = 8
        print ("batchSize")
        print (batchSize)
        #NN Architecture File Name 
        writeFile = current_dir + "embeddings"+"/t_"
        writeFile += filename
        print("writeFile")
        print(writeFile)
        myFile = open(writeFile, 'w')
        with myFile:    
            header = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','training_loss','root_mean_squared_error', 
                      'val_loss', 'val_root_mean_squared_error', 'test_loss' , 'test_acc']
            writer = csv.DictWriter(myFile, fieldnames=header)    
            writer.writeheader()

            # Create a NN with one input layer, 1 dense layers and 1 output layer
            n_layers = 1
            layers = []
            layers.append(Dense(int(5), activation=tf.nn.leaky_relu))
            layers.append(Dense(int(1), activation='sigmoid'))
              # Instantiate a simple classification model
            model = tf.keras.Sequential(layers)
            # Instantiate a logistic loss function that expects integer targets.
            lossfuntion = tf.keras.losses.MeanSquaredError()
            # Instantiate an accuracy metric.
            accuracyType = tf.keras.metrics.RootMeanSquaredError()
            # Instantiate an optimizer.
            learning_rate = 0.001
            optimizer = tf.keras.optimizers.Adam(learning_rate)
            filepath=current_dir+"embeddings/" +"emb." #modelFile+"_"+"emb."
            checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max') #monitor='val_tpr',
            callbacks_list_NN = [checkpoint]
            #First, call compile to configure the optimizer, loss, and metrics to monitor.
            model.compile(optimizer=optimizer, loss=lossfuntion, metrics=[accuracyType])
            history = model.fit(X_train, y_train, validation_split=0.1,  callbacks=callbacks_list_NN, verbose=2, epochs=1000, batch_size=batchSize)
            #history = model.fit(X_train, y_train, validation_split=0.1, verbose=2, epochs=1000, batch_size=batchSize)
            print (history.history)
            training_loss = history.history["loss"]
            root_mean_squared_error = history.history["root_mean_squared_error"]
            val_loss = history.history["val_loss"]
            val_root_mean_squared_error = history.history["val_root_mean_squared_error"]
            test_loss, test_acc = model.evaluate(X_test, y_test)
            print('####################################################################################')
            print('Test accuracy:', test_acc)
            print('####################################################################################')
            # save into a csv file
            writer.writerow({ 'F1':f1,'F2':f2,'F3':f3,'F4': f4,'F5':f5,
                               'F6':f6,'F7':f7,'F8':f8,'F9':f9,'F10':f10,
                               'training_loss' : training_loss[-1], 
                               'root_mean_squared_error' : root_mean_squared_error[-1], 
                               'val_loss' : val_loss[-1], 
                               'val_root_mean_squared_error' : val_root_mean_squared_error[-1], 
                               'test_loss': test_loss,
                               'test_acc' : test_acc,})
        myFile.close()