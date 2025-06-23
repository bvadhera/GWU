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
#from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input,Dense,Lambda,Concatenate 
from keras.callbacks import ModelCheckpoint
import keras.backend as K
#from tensorflow.python.keras.backend import set_session
#from tensorflow.python.keras.models import load_model

# Helper Files
import actorCriticAgent_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained

import helperFunctions_JustAccuracyLamdaForwardLookupTreeActorCriticTrained
import randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained
import reinforcementTraining_JustAccuracyTrainedLamdaForwardLookupTreeActorCriticTrained 

#tf.compat.v1.disable_eager_execution()
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)
tf.random.set_seed(545654)


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
    weight = 0.1  
    acc_loss =  K.mean(K.square(y_true[:,:1] - y_pred[:,:1]))
    return acc_loss

 
 
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


 
 


#
#   CODE FOR NETWORK ONE without RL
#   
# Dense Layers nodes for encoder and decoder
maxNumOfLayers = 3.0
minNumOfLayers = 0.0 
latent_dim = 10
latent_dim1 = 15
latent_dim11 = 20
latent_dim2 = 25
divideBy = 24.0
multiplyBy = 3.0
#os.chdir("/home/bvadhera/huber/")
cwd = os.getcwd()
#current_dir  = "/home/bvadhera/huber/"
current_dir  = cwd +"/"

print (current_dir)
# load The TrainingDataSet File
 
accuracy_DataSet = current_dir+"testDataForAccuracy.csv"
 
# Open a csv reader called DictReader
# iterate over each line as a ordered dictionary and print only few column by column name
df = pd.read_csv(accuracy_DataSet)
print(df.head())
testDataToCompareAccuracies = df
normalizeProp = list(df.columns.values)
trainProperties = list(df.columns.values)
# remove 'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10'
#randomNetworkGeneration_UpdatedAccuracyModel.removeOneHotColumns(normalizeProp)
#Also remove 'test_acc' so that we dont normalize the accuracy
normalizeProp.remove('test_acc')
randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.removeOneHotColumns(trainProperties)
print("normalizeProp")
print(normalizeProp)
print("trainProperties")
print(trainProperties)
trainProperties.remove('x')
trainProperties.remove('y')
 
y = df[trainProperties]
X = df[normalizeProp]

 
print(y.head())

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


inputs_accuracy_model=Input((12,))
#Hidden Layers
hiddenP_layer_1 = Dense(int(15),name='accuracy_network_Input', activation=tf.nn.leaky_relu)
hiddenP_layer_11 = Dense(int(20), activation=tf.nn.leaky_relu)
#print ('hiddenP_layer_1.shape:', hiddenP_layer_1.shape)
hiddenP_layer_2 = Dense(int(25), activation=tf.nn.leaky_relu)
#print ('hiddenP_layer_2.shape:', hiddenP_layer_2.shape)
#Call Model to get predicted Accuracy as output
accuracy_layer = Dense(int(1),name='accuracy_network', activation='sigmoid')
#print ('accuracy_layer.shape:', accuracy_layer.shape)

accuracy_model = Model(inputs=inputs_accuracy_model, 
                            outputs=[accuracy_layer(hiddenP_layer_2(hiddenP_layer_11(hiddenP_layer_1(inputs_accuracy_model))))])
#accuracy_model.compile(optimizer=optimizer, loss=None,metrics=None)

accuracy_model.compile(optimizer=optimizer, loss={'accuracy_network': custom_loss_accuracy},
                     metrics={'accuracy_network': [ mean_sqe_pred]}) #, run_eagerly=True)

filepath_acc=current_dir +"acc_acc/" #modelFile+"_"+"acc"
checkpoint1 = ModelCheckpoint(filepath_acc, monitor='val_loss', verbose=1, save_best_only=True, mode='max') #monitor='val_tpr',
callbacks_list_NN_Acc = [checkpoint1]
###### Huber Removed here RL 
#history_accuracy_model = accuracy_model.fit(X_train, {'accuracy_network': y_train[['test_acc']]}, validation_split=0.1,  
#                                                                   callbacks=callbacks_list_NN_Acc, verbose=2, epochs=7000, batch_size=batchSize)

#accuracy_model.save_weights(filepath_acc + "weights")

#print (history.history)
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

#print (type(y_train))

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
accuracy_model.load_weights(filepath_acc + "weights")
#### pick one valid generated network after normalization 
# line 184 in spreedsheet with accuracy 0.8900804289544236
num_layers,num_node1,num_node2 = randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.getRandomLegalNetwork(divideBy,maxNumOfLayers, minNumOfLayers ) # generate valid normalize network
test_nn = pd.DataFrame({'F1': [0], 'F2': [1],'F3': [0], 'F4': [0],'F5': [0],
                    'F6': [0],'F7': [0], 'F8': [0], 'F9': [0],'F10': [0], 'num_layers_N': [num_layers],
                    'num_node1_N': [num_node1],'num_node2_N': [num_node2]}, index=['50112'])
print("test_nn_fabricated.head() ")
print(test_nn.head())
test_nn_rl = test_nn
print('####################################################################################')
grid_DataSet = current_dir+"gridpoints_160.csv"
#randomNetworkGeneration_UpdatedAccuracyModel.generateEmbeddingsAccuracy(model,grid_DataSet,current_dir)
print('# Test Accuracy For Hundred NN after training#') # Error generate with new formula and normalizaton
#randomNetworkGeneration_UpdatedAccuracyModel.testAccuracyForListOfNN_Mix(model, 500, divideBy, max_accuracy_before_norm, min_accuracy_before_norm, maxNumOfLayers, minNumOfLayers,current_dir)
#randomNetworkGeneration_UpdatedAccuracyModel.testAccuracyForListOfNNLegal(model, 500, divideBy, max_accuracy_before_norm, min_accuracy_before_norm, maxNumOfLayers, minNumOfLayers,24,current_dir )
print('####################################################################################')
#randomNetworkGeneration_UpdatedAccuracyModel.testAccuracyForTrainedListOfNN_Acc(model,testDataToCompareAccuracies,current_dir)
print('####################################################################################')
print('# Train for actor-critic #')
print('####################################################################################')


agent = actorCriticAgent_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.Agent()

#agent = reinforcementTraining_JustAccuracyTrainedLamdaForwardLookupTreeActorCriticTrained.rlTraining(test_nn_rl, model, agent, divideBy, maxNumOfLayers, minNumOfLayers,current_dir)
 

############################################################################
## Testing the Policy
# This is for testing after tarining is done.
# test for network 2,10,10 expecting accuracy 1
# Which is 0.08, 0.4, 0.4 and accuracy = 1
total_reward = 0
#-----------------
# If you want to directly load the agent weights the uncomment followings
        # Save weights in a file
agent.actor_main.load_weights(filepath + "/random_noHack/" + "actor_main_weights")
agent.actor_target.load_weights(filepath + "/random_noHack/" + "actor_target_weights")
agent.critic_main.load_weights(filepath + "/random_noHack/" + "critic_main_weights")
agent.critic_main2.load_weights(filepath + "/random_noHack/" + "critic_main2_weights")
agent.critic_target.load_weights(filepath + "/random_noHack/" + "critic_target_weights")
agent.critic_target2.load_weights(filepath + "/random_noHack/" + "critic_target2_weights")


#randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.generateEmbeddingsCritic(grid_DataSet, agent,current_dir)
randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.generateEmbeddingsActor(grid_DataSet, agent,current_dir)
#randomNetworkGeneration_UpdatedAccuracyModel.generateEmbeddingsActorNew(grid_DataSet, agent,current_dir)   
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
        encoded_x, encoded_y,n_layers,num_node1,num_node2, prev_accuracy,reward,total_reward,qValue1, qValue2, qActions_1, qActions_2, qActions_3= helperFunctions_JustAccuracyLamdaForwardLookupTreeActorCriticTrained.getPolicyoutput(index1,model,agent,divideBy,maxNumOfLayers, minNumOfLayers, decoder_model,current_dir,file_utility_values_R)
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
    encoded_x, encoded_y,n_layers,num_node1,num_node2, prev_accuracy,reward,total_reward,qValue1,qValue2,qActions_1, qActions_2,qActions_3 = helperFunctions_JustAccuracyLamdaForwardLookupTreeActorCriticTrained.getPolicyoutput(0,model,agent,divideBy,maxNumOfLayers, minNumOfLayers, decoder_model,current_dir,file_utility_values_R )
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

