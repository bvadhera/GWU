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

# Helper Files
import actorCriticAgent_adam_0001
import helperFunctions
import randomNetworkGeneration
import reinforcementTraining_adam_0001

#tf.compat.v1.disable_eager_execution()
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)
tf.random.set_seed(934733)


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu,True)


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

 
# Accuracy Matrics First  ERROR ( Need to fix as per huber as we have in between boundry conditions like 1.5 etc)
def decoder_accuracy(y_true,y_pred):
    #predx = K.batch_get_value(y_pred[:,:3])  # tensor for num_layers, node1 & node2 after decoder
    #truex = K.batch_get_value(y_true[:,:3]) # tensor for num_layers, node1 & node2 before decoder
    
    #feature_name*(maxNumOfLayers - minNumOfLayers) + minNumOfLayers
    #Now here minNumOfLayers = 0 
    # maxNumOfLayers - minNumOfLayers = 2
    matrix_multiply = np.array([(multiplyBy, divideBy, divideBy)],dtype = 'float32')
    matrix_multiply = tf.constant(matrix_multiply)
    denorm_y_true = tf.multiply(y_true[:,1:4],  matrix_multiply) 
    denorm_y_pred = tf.multiply(y_pred[:,1:4],  matrix_multiply) 
    
    #denorm_y_true = y_true[:,1:4] * divideBy 
    #denorm_y_pred = y_pred[:,1:4] * divideBy 

    diff = K.equal(K.mean(K.round(denorm_y_true - denorm_y_pred),  axis=-1), 0)
    return(K.cast(diff,tf.float32))



def testencoderdecoderForTrainedListOfNN():
    listOfNN_WithAcc = testDataToCompareAccuracies.values.tolist()
    encoderDecoderFile = current_dir +"38234EpisodsRandomRLRandomPolicyRandom100NN/encoderDecoderTest.csv"
    myEncoderDecoderFile = open(encoderDecoderFile, 'w')
    ##  TODO - CHange the headings as per the results
    headerEncoderDecoder = ['elayers','enode1', 'enode2', 'dlayers','dnode1', 'dnode2'] 
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



def testdecoderForTrainedListOfNN():
    listOfNN_WithAcc = [[-1.4447945,-7.0156035],[-1.4947945,-6.9656034],[-0.72571605,-4.1927166],[-0.77571607,-4.242717],
                        [2.9229288,-3.7564206],[2.8729289,-3.8064206],[-3.406248,-5.3279796],[-3.404458,-5.33835],
                        [-0.6802437,-2.86686],[-0.7302437,-2.9168599]]
    decoderFile = current_dir+"38234EpisodsRandomRLRandomPolicyRandom100NN/DecoderTest.csv"
    myDecoderFile = open(decoderFile, 'w')
    ##  TODO - CHange the headings as per the results
    headerDecoder = ['enode1', 'enode2', 'dlayers','dnode1', 'dnode2'] 
    writerDecoder = csv.DictWriter(myDecoderFile, fieldnames=headerDecoder) 
    writerDecoder.writeheader()
    for j in listOfNN_WithAcc:

        ## Test decodings and Decode those embeddings
        embeddings = np.array([j])
        decodedOutput = decoder_model(embeddings)
        print ("decodedOutput")
        print (decodedOutput)
        # get the x & y points from concat_output
        decodedOutput_array = np.asarray(decodedOutput)
        dlayers = decodedOutput_array.item(0)
        dnode1 = decodedOutput_array.item(1)
        dnode2 = decodedOutput_array.item(2)
        # save into a csv file
        writerDecoder.writerow({ 
                'enode1' : embeddings[0][0],
                'enode2' : embeddings[0][1],
                'dlayers' : dlayers, 
                'dnode1' : dnode1,
                'dnode2' : dnode2
                })
    myDecoderFile.close()


#
#   CODE FOR NETWORK ONE without RL
#   
# Dense Layers nodes for encoder and decoder

latent_dim = 10
latent_dim1 = 15
latent_dim11 = 20
latent_dim2 = 25
divideBy = 0.0
#os.chdir("/home/bvadhera/huber/")
cwd = os.getcwd()
#current_dir  = "/home/bvadhera/huber/"
current_dir  = cwd +"/"

print (current_dir)
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
randomNetworkGeneration.removeOneHotColumns(normalizeProp)
#Also remove 'test_acc' so that we dont normalize the accuracy
normalizeProp.remove('test_acc')
print("normalizeProp")
print(normalizeProp)
divideBy = helperFunctions.getMaxNumOfNodesInDataSet(df) - helperFunctions.getMinNumOfNodesInDataSet(df)
multiplyBy = maxNumOfLayers - minNumOfLayers
#Normalize the data for num_layers, num_node1, num_node2 only
helperFunctions.normalize(df,normalizeProp,divideBy, minNumOfLayers, maxNumOfLayers)
print("df.head() after normalized data")
print(df.head())
origProperties = list(df.columns.values)
trueProperties = list(df.columns.values)
# Now save two dataframes one for final test which is y (num_layers_N  num_node1_N  num_node2_N, test_acc)
# and another for X (num_layers_N, num_node1_N, num_node2_N,,F1,F2,F3,F4,F5,F6,F7,F8,F9,F10)

# remove properties 'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'
#  and 'num_layers', 'num_node1', 'num_node2'
randomNetworkGeneration.removeOneHotColumns(trueProperties)
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

num_layers,num_node1,num_node2 = randomNetworkGeneration.getRandomLegalNetwork(divideBy,maxNumOfLayers, minNumOfLayers ) # generate valid normalize network
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
        concat_output = randomNetworkGeneration.getConcatinateLayerOutput(test_nn, model)
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
#testdecoderForTrainedListOfNN()
#testencoderdecoderForTrainedListOfNN()
#randomNetworkGeneration.generateEmbeddingsAccuracy(model,grid_DataSet,current_dir)

print('# Test Accuracy For Hundred NN after training#') # Error generate with new formula and normalizaton

#randomNetworkGeneration.testAccuracyForListOfNN_Mix(model, 500, divideBy, max_accuracy_before_norm, min_accuracy_before_norm, maxNumOfLayers, minNumOfLayers,current_dir)

#randomNetworkGeneration.testAccuracyForListOfNNLegal(model, 500, divideBy, max_accuracy_before_norm, min_accuracy_before_norm, maxNumOfLayers, minNumOfLayers,24,current_dir )

print('####################################################################################')

#randomNetworkGeneration.testAccuracyForTrainedListOfNN(model,testDataToCompareAccuracies,current_dir)

print('####################################################################################')
print('# Train for actor-critic #')
print('####################################################################################')

agent = actorCriticAgent_adam_0001.Agent()

agent = reinforcementTraining_adam_0001.rlTraining(test_nn_rl, model, agent, divideBy, maxNumOfLayers, minNumOfLayers,current_dir)
 








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


#randomNetworkGeneration.generateEmbeddingsCritic(grid_DataSet, agent,current_dir)
#randomNetworkGeneration.generateEmbeddingsActor(grid_DataSet, agent,current_dir)
#randomNetworkGeneration.generateEmbeddingsActorNew(grid_DataSet, agent,current_dir)   
#-------------------
random100 = True  # Use random NN of 100 NN to do policy evaluation
writePolicyOutPutFile = current_dir+"38234EpisodsRandomRLRandomPolicyRandom100NN/policyOutPutFor.csv"
myPolicyOutPutFile = open(writePolicyOutPutFile, 'w')
##  TODO - CHange the headings as per the results
headerPolicyOutPutFile = ['encoded_x','encoded_y', 'layers', 'nnode1','nnode2','accuracy','reward','total_reward','critic_value','actor_value_1','actor_value_2'] 
writerPolicyOutPut = csv.DictWriter(myPolicyOutPutFile, fieldnames=headerPolicyOutPutFile) 
writerPolicyOutPut.writeheader()

if random100:
    #=========
    for index1 in range(100):
        sameStartNode = False
        encoded_x, encoded_y,n_layers,num_node1,num_node2, prev_accuracy,reward,total_reward,qValue,qActions_1, qActions_2 = helperFunctions.getPolicyoutput(index1,model,agent,divideBy,maxNumOfLayers, minNumOfLayers, decoder_model,current_dir)
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
                'actor_value_2' : qActions_2
                })
        state, num_layers,num_node1,num_node2, accuracy_state = randomNetworkGeneration.getStateAndAccuracy(model,divideBy,maxNumOfLayers, minNumOfLayers)
else:
    #  same as testNN
    sameStartNode = True                     
    encoded_x, encoded_y,n_layers,num_node1,num_node2, prev_accuracy,reward,total_reward,qValue,qActions_1, qActions_2 = helperFunctions.getPolicyoutput(0,model,agent,divideBy,maxNumOfLayers, minNumOfLayers, decoder_model )
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

