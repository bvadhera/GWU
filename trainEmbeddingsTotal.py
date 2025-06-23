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



def normalize(df_,features):
    #result = df.copy()
    for feature_name in features:
        max_value = df_[feature_name].max()
        min_value = df_[feature_name].min() ;print("min =",min_value,"\t max=",max_value)
        df_[feature_name+"_N"] = (df[feature_name] - min_value) / (max_value - min_value)



current_dir  = "/home/bvadhera/huber/"
inputEmbeddingsFile = "inputEmbeddingsTotal.csv"
print("inputFile")
print (inputEmbeddingsFile)
writeFile = current_dir + "embeddings"+"/trained_total"
writeFile += inputEmbeddingsFile
myFile = open(writeFile, 'w')
header = ['training_loss','root_mean_squared_error', 
                      'val_loss', 'val_root_mean_squared_error', 'test_loss' , 'test_acc']  
writer = csv.DictWriter(myFile, fieldnames=header) 
writer.writeheader()
print("writeFile")
print(writeFile)
#For each Dataset File create test and train data
df = pd.read_csv(current_dir+"embeddings/"+inputEmbeddingsFile)
print(df.head())
x_properties = list(df.columns.values)

print(x_properties)
x_properties.remove('F1')
x_properties.remove('F2')
x_properties.remove('F3')
x_properties.remove('F4')
x_properties.remove('F5')
x_properties.remove('F6')
x_properties.remove('F7')
x_properties.remove('F8')
x_properties.remove('F9')
x_properties.remove('F10')

X_before = df[x_properties]
y_properties = list(X_before.columns.values)
x_properties.remove('test_acc')
y_properties.remove('x')
y_properties.remove('y')
print(x_properties)
print(y_properties)
X = X_before[x_properties]
print(y_properties)
y = X_before[y_properties]

print("X.head()")
print(X.head())
print("y.head()")
print(y.head())

# Divide the data into test and train sets
a = random.randint(1,50)
print ("random_state")
print (a)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=a)
print("The type of x_train is : ",type(X_train))
# Get Batch size for X_train
batchSize = 32
print ("batchSize")
print (batchSize)   

# Create a NN with one input layer, 1 dense layers and 1 output layer
n_layers = 2
layers = []
layers.append(Dense(int(7), activation=tf.nn.leaky_relu))
layers.append(Dense(int(3), activation=tf.nn.leaky_relu))
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
writer.writerow({ 'training_loss' : training_loss[-1], 
                  'root_mean_squared_error' : root_mean_squared_error[-1], 
                  'val_loss' : val_loss[-1], 
                  'val_root_mean_squared_error' : val_root_mean_squared_error[-1], 
                  'test_loss': test_loss,
                  'test_acc' : test_acc})
myFile.close()