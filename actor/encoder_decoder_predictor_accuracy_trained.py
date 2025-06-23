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

#tf.compat.v1.disable_eager_execution()
tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)
tf.random.set_seed(934733)
avg_rewards_list = []

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu,True)

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
    weight = 0.1  
    loss =  K.mean(K.square( y_true[:,1:4] - y_pred[:,1:4]), axis=-1)  + weight*(K.square(y_true[:,:1] - y_pred[:,:1]))
    return loss

def custom_loss_actor(y_true, y_pred):
    print("type(y_true)")
    print(type(y_true) )
    print("type(y_pred)")
    print(type(y_pred) )
    # convert y_true to tensorflow.python.keras.engine.keras_tensor.KerasTensor
    print ('##################################')  # shape 
    print ('y_true.shape:', y_true.shape)  # shape 
    print ('y_pred.shape:',y_pred.shape)   # shape 
    # Return a function
    return 0*(K.mean(K.square( y_true - y_pred), axis=-1) )

def custom_loss_critic(y_true, y_pred):
    print("type(y_true)")
    print(type(y_true) )
    print("type(y_pred)")
    print(type(y_pred) )
    # convert y_true to tensorflow.python.keras.engine.keras_tensor.KerasTensor
    print ('##################################')  # shape 
    print ('y_true.shape:', y_true.shape)  # shape 
    print ('y_pred.shape:',y_pred.shape)   # shape 
    # Return a function
    return 0*(K.mean(K.square( y_true - y_pred), axis=-1) )
 
# Accuracy Matrics First  ERROR ( Need to fix as per huber as we have in between boundry conditions like 1.5 etc)
def decoder_accuracy(y_true,y_pred):
    #predx = K.batch_get_value(y_pred[:,:3])  # tensor for num_layers, node1 & node2 after decoder
    #truex = K.batch_get_value(y_true[:,:3]) # tensor for num_layers, node1 & node2 before decoder
    
    #feature_name*(maxNumOfLayers - minNumOfLayers) + minNumOfLayers
    #Now here minNumOfLayers = 0 
    # maxNumOfLayers - minNumOfLayers = 2
    matrix_multiply = np.array([(multiplyBy,divideBy,divideBy)],dtype = 'float32')
    matrix_multiply = tf.constant(matrix_multiply)
    denorm_y_true = tf.multiply(y_true[:,1:4],  matrix_multiply) 
    denorm_y_pred = tf.multiply(y_pred[:,1:4],  matrix_multiply) 
    
    #denorm_y_true = y_true[:,1:4] * divideBy 
    #denorm_y_pred = y_pred[:,1:4] * divideBy 

    diff = K.equal(K.mean(K.round(denorm_y_true - denorm_y_pred),  axis=-1), 0)
    return(K.cast(diff,tf.float32))

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

# Accuracy Matrics First
def actor_accuracy(y_true,y_pred):
    return(K.zeros(1))

# Accuracy Matrics First
def critic_accuracy(y_true,y_pred):
    return(K.zeros(1))

def getMaxNumOfNodesInDataSet(df_max):
    mLayers = df_max['num_layers'].max()
    #valid layers are only 2
    mLayers = 2
    # iterate as many layers and create node columns 
    i = 0
    maxNumOfNodes = []
    for i in range(int(mLayers)):
      node = "num_node" + str(i+1)
      attrib = df_max[node]
      maxNumOfNodes.append(int(attrib.max()))
    print (max(maxNumOfNodes))
    #get max number of nodes
    return max(maxNumOfNodes)

def getMinNumOfNodesInDataSet(df_max):
    mLayers = df_max['num_layers'].max()
    #valid layers are only 2
    mLayers = 2
    # iterate as many layers and create node columns 
    i = 0
    minNumOfNodes = []
    for i in range(int(mLayers)):
      node = "num_node" + str(i+1)
      attrib = df_max[node]
      minNumOfNodes.append(int(attrib.min()))
    print (min(minNumOfNodes))
    #get max number of nodes
    return min(minNumOfNodes)

def addPaddingofZero(df):
    df["actor_X"] = 0.0
    df["actor_Y"] = 0.0
    df["critic_Value"] = 0.0


def normalize(df_,features, divideBy, minNumOfLayers, maxNumOfLayers):
  print(features)
  for feature_name in features:
    if (feature_name == 'num_layers'):  #  num_layers - 1 / 2-1   so that we get 0 or 1 
        df_[feature_name+"_N"] = (df_[feature_name] - minNumOfLayers)/(maxNumOfLayers - minNumOfLayers)
    else: 
        df_[feature_name+"_N"] = df_[feature_name]/divideBy # 

def dNormalize(df_,features, divideBy, minNumOfLayers, maxNumOfLayers):
  print(features)
  for feature_name in features: 
    if (feature_name == 'num_layers'):
        df_[feature_name+"_N"] = (df_[feature_name] + minNumOfLayers)*(maxNumOfLayers - minNumOfLayers) 
    else:
        df_[feature_name+"_N"] = df_[feature_name]*divideBy
 
def getStateAndAccuracy(model, divideBy,maxNumOfLayers, minNumOfLayers):
    #Choose randon network to find the best network
    num_layers,num_node1,num_node2 = getRandomLegalNetwork(divideBy,maxNumOfLayers, minNumOfLayers )
    test_nn = pd.DataFrame({'F1': [0], 'F2': [1],'F3': [0], 'F4': [0],'F5': [0],
                'F6': [0],'F7': [0], 'F8': [0], 'F9': [0],'F10': [0], 'num_layers_N': [num_layers],
                'num_node1_N': [num_node1],'num_node2_N': [num_node2]}, index=['501'])
    print("test_nn_fabricated.head() ")
    print(test_nn.head())
    concat_output = getConcatinateLayerOutput(test_nn, model)
    derived_concat_output =  concat_output[0][0]
    accuracy_state = getAccuracyLayerOutput(concat_output, model)
    state = derived_concat_output
    return state, num_layers,num_node1,num_node2, accuracy_state

def decodeEncodedEmbeddings(state):
    #my_list = np.array([0.00000000309866918541956,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    
    state = list(state) 

    #state = list(my_list)
    
    state = tf.convert_to_tensor([state], dtype=tf.float32)
    split_state_for_encoded_Values = Lambda(lambda x: tf.split(x,[2,10],axis=1))(state)
    encoded_network = split_state_for_encoded_Values[0]
    encoded_arr = tf.keras.backend.get_value(encoded_network)
    #print ("encoded_output")
    #print (encoded_arr[0][0])
    #print (encoded_arr[0][1])
    # send them for decoding
    decod_output = decoder_model(encoded_network)
    # get the x & y points from concat_output
    decod_output_array = np.asarray(decod_output)

    nLayers = decod_output_array.item(0)
    numNode1 = decod_output_array.item(1)
    numNode2 = decod_output_array.item(2)
    print ("decoded_output")
    print (nLayers)
    print (numNode1)
    print (numNode2)
    if ((nLayers< 0) or (numNode1< 0) or (numNode2 < 0)):
        print ("decod_output is wrong - Investigate")
        print (n_layers, num_node1, num_node2)
    return encoded_arr[0][0], encoded_arr[0][1], nLayers, numNode1, numNode2

def getPolicyoutput(index):
    
    total_reward = 0
    max_steps_per_episode = 1000 # trejectories 1000
    alpha = 0.05
   
    #file_accuracy_values_R.write(str(prev_accuracy) + " - prev_accuracy\n")
    done = False
    steps_per_episode = 0
    qValue = 0.0
    # This is for testing after tarining is done.
    # test for network 2,10,10 expecting accuracy 1
    writeTrejectoriesFile = "/home/bvadhera/huber/38234EpisodsRandomRLRandomPolicyRandom100NN/policyTrejectories" + str(index) +".csv"
    myTrejectoriesFile = open(writeTrejectoriesFile, 'w')
    ##  TODO - CHange the headings as per the results
    headerTrejectories = ['encoded_x','encoded_y', 'layers', 'nnode1','nnode2','accuracy','reward','total_reward','critic_value','actor_value_1','actor_value_2'] 
    writerTrejectories = csv.DictWriter(myTrejectoriesFile, fieldnames=headerTrejectories) 
    writerTrejectories.writeheader()

    print("writerTrejectories")
    print(writerTrejectories)
    state, num_layers,num_node1,num_node2, accuracy_state = getStateAndAccuracy(model,divideBy,maxNumOfLayers, minNumOfLayers)
    writerTrejectories.writerow({ 'encoded_x' : state[0], 
                'encoded_y' : state[1], 
                'layers' : num_layers,
                'nnode1' : num_node1,
                'nnode2' : num_node2,
                'accuracy' : accuracy_state, 
                'reward': 0.0,
                'total_reward': total_reward,
                'critic_value' : 0.0,
                'actor_value_1'  : 0.0,
                'actor_value_2'  : 0.0
                })
    prev_accuracy = (accuracy_state[0][0][0]).item()

    while not done:
            action = agent.act(state, True)
            # Call Critic
            # ERROR HUBER
            # convert states and new states, rewards and actions to tensor.
            states = np.array([state])
            actions = np.array([action])
            states = tf.convert_to_tensor(states, dtype= tf.float32)
            actions = tf.convert_to_tensor(actions, dtype= tf.float32)
            qValue = tf.squeeze(agent.critic_main(states, actions), 1)
            qValue = (tf.keras.backend.get_value(qValue))[0]
            #get actor value
            #actions = agent.actor_main(states)
            qActions_1 = (tf.keras.backend.get_value(action))[0][0]
            qActions_2 = (tf.keras.backend.get_value(action))[0][1]
            next_state, next_accuracy, reward, done, steps_per_episode  = returnStepValues(action, alpha, state, prev_accuracy, steps_per_episode,max_steps_per_episode, model, done)   
            # Function to  next_state = state + alpha * action
            # reward  = diff between the accuracy of (next state and state)
            # done = when is when we acceed 20 steps
            
            # Save the current state of the environment to the new state
            state = next_state
            prev_accuracy = next_accuracy
            total_reward += reward
            encoded_x, encoded_y,nlayers,numnode1,numnode2 = decodeEncodedEmbeddings(state)
            layers = nlayers
            nnode1 = numnode1
            nnode2 = numnode2
            if ((layers < 0) or (nnode1 < 0) or (nnode2 < 0)):
                print ("decod_output is wrong - Investigate")
            # save into a csv file
            writerTrejectories.writerow({ 'encoded_x' : encoded_x, 
                  'encoded_y' : encoded_y, 
                  'layers' : layers,
                  'nnode1' : nnode1,
                  'nnode2' : nnode2,
                  'accuracy' : prev_accuracy, 
                  'reward': reward,
                  'total_reward': total_reward,
                  'critic_value' : qValue,
                  'actor_value_1' : qActions_1,            
                  'actor_value_2'  : qActions_2
                  })
            # decode and save rewards
            if done:
                print(total_reward)
                myTrejectoriesFile.close()
                encoded_x, encoded_y,nlayers,numnode1,numnode2 = decodeEncodedEmbeddings(state) 
                layers = nlayers
                nnode1 = numnode1
                nnode2 = numnode2
                return (encoded_x, encoded_y,layers,nnode1,nnode2, prev_accuracy,reward,total_reward,qValue,qActions_1,qActions_2)
                


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

def getConcatinateLayerOutput(test_nn, model):
    concat_func = K.function([model.layers[0].input],
                                [model.get_layer('concatenate').output])
    concat_output = concat_func([test_nn])
    print (concat_output)
    return concat_output  

def getAccuracyLayerOutput(concat_output, model):
    accuracy_func = K.function([model.get_layer('dense_8').input],
                                [model.get_layer('dense_11').output])
    accuracy_output = accuracy_func([concat_output])
    print (accuracy_output)
    return accuracy_output  
 
     
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



#
#   CODE FOR NETWORK ONE without RL
#   
# Dense Layers nodes for encoder and decoder

latent_dim = 10
latent_dim1 = 15
latent_dim11 = 20
latent_dim2 = 25
divideBy = 0.0
 
cwd = os.getcwd()
#current_dir  = "/home/bvadhera/huber/"
current_dir  = cwd +"/"


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
removeOneHotColumns(normalizeProp)
#Also remove 'test_acc' so that we dont normalize the accuracy
normalizeProp.remove('test_acc')
print("normalizeProp")
print(normalizeProp)
divideBy = getMaxNumOfNodesInDataSet(df) - getMinNumOfNodesInDataSet(df)
multiplyBy = maxNumOfLayers - minNumOfLayers
#Normalize the data for num_layers, num_node1, num_node2 only
normalize(df,normalizeProp,divideBy, minNumOfLayers, maxNumOfLayers)
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
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min') #monitor='val_tpr',
callbacks_list_NN = [checkpoint]
#First, call compile to configure the optimizer, loss, and metrics to monitor. 
###### Huber Removed here RL  
model.compile(optimizer=optimizer, loss={'network_with_accuracy': custom_loss},
                     metrics={'network_with_accuracy': [decoder_accuracy , mean_sqe_pred, mean_sqe_pred_legal]}) #, run_eagerly=True)

print (type(y_train))

###### Huber Removed here RL 
history = model.fit(X_train, {'network_with_accuracy': y_train[['test_acc', 'num_layers_N', 'num_node1_N', 'num_node2_N']] }, validation_split=0.1,  
                                                                   callbacks=callbacks_list_NN, verbose=2, epochs=7000, batch_size=batchSize)#print (history.history)


# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
#model.save(current_dir+"my_model.h5") 
# Save the weights Manually
model.save_weights(filepath + "weights")

# Restore the weights
#model.load_weights(filepath + "weights")
# Recreate the exact same model, including its weights and the optimizer
#new_model = tf.keras.models.load_model('my_model.h5')


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

num_layers,num_node1,num_node2 = getRandomLegalNetwork(divideBy,maxNumOfLayers, minNumOfLayers ) # generate valid normalize network
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
        concat_output = getConcatinateLayerOutput(test_nn, model)
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
                 
myFile.close()
print('####################################################################################')

'''
grid_DataSet = current_dir+"gridpoints_160.csv"
#testencoderdecoderForTrainedListOfNN()
#generateEmbeddingsAccuracy(model,grid_DataSet)

print('# Test Accuracy For Hundred NN after training#') # Error generate with new formula and normalizaton

#testAccuracyForListOfNN_Mix(model, 500, divideBy, max_accuracy_before_norm, min_accuracy_before_norm, maxNumOfLayers, minNumOfLayers)

#testAccuracyForListOfNNLegal(model, 500, divideBy, max_accuracy_before_norm, min_accuracy_before_norm, maxNumOfLayers, minNumOfLayers,24 )

print('####################################################################################')

#testAccuracyForTrainedListOfNN(model,testDataToCompareAccuracies)

print('####################################################################################')
print('# Train for actor-critic #')
print('####################################################################################')

agent = Agent()


# Now train actor critic
# The main program, Main loop 
## Phase I use previos test network for every episode.
with tf.device('GPU:0'):
    not_random = False
    if not_random:  # RUN WITH FIXED NETWORK

        # Load weights
        #agent.actor_main.load_weights(filepath + "/non_random/" + "actor_main_weights")
        #agent.actor_target.load_weights(filepath + "/non_random/" + "actor_target_weights")
        #agent.critic_main.load_weights(filepath + "/non_random/" + "critic_main_weights")
        #agent.critic_main2.load_weights(filepath + "/non_random/" + "critic_main2_weights")
        #agent.critic_target.load_weights(filepath + "/non_random/" + "critic_target_weights")
        #agent.critic_target2.load_weights(filepath + "/non_random/" + "critic_target2_weights")

        #tf.config.experimental.reset_memory_stats('GPU:0')
        file_totalReward = open('/home/bvadhera/huber/rl_results/totalReward.csv', 'w')
        file_avgReward = open('/home/bvadhera/huber/rl_results/avgReward.csv', 'w')
        file_reward = open('/home/bvadhera/huber/rl_results/reward.csv', 'w')
        file_actions = open('/home/bvadhera/huber/rl_results/actions.csv', 'w')
        file_actor_loss = open('/home/bvadhera/huber/rl_results/actor_loss.csv', 'w')   
        file_critic_value = open('/home/bvadhera/huber/rl_results/critic_value.csv', 'w')  
        file_critic_value2 = open('/home/bvadhera/huber/rl_results/critic_value2.csv', 'w') 
        file_critic_loss = open('/home/bvadhera/huber/rl_results/critic_loss.csv', 'w')  
        file_critic_loss2 = open('/home/bvadhera/huber/rl_results/critic_loss2.csv', 'w')
        file_new_policy_actions  = open('/home/bvadhera/huber/rl_results/new_policy_actions.csv', 'w') 
        file_target_actions = open('/home/bvadhera/huber/rl_results/target_actions.csv', 'w')
        file_target_next_state_values = open('/home/bvadhera/huber/rl_results/target_next_state_values.csv', 'w')
        file_target_next_state_values2 = open('/home/bvadhera/huber/rl_results/target_next_state_values2.csv', 'w')
        file_next_state_target_value = open('/home/bvadhera/huber/rl_results/next_state_target_value.csv', 'w')
        file_target_values = open('/home/bvadhera/huber/rl_results/target_values.csv', 'w')
        file_state_values = open('/home/bvadhera/huber/rl_results/state_values.csv', 'w')
        file_accuracy_values = open('/home/bvadhera/huber/rl_results/accuracy_values.csv', 'w')
    
       
        #episods = 100  
        #episods = 2000
        episods = 10000
        ep_reward = []
        total_avgr = []
        target = False
        max_steps_per_episode = 300 # trejectories
        alpha = 0.05
        # Take Valid Network
        test_nn = test_nn_rl
        concat_output = getConcatinateLayerOutput(test_nn, model)
        derived_concat_output =  concat_output[0][0]
        accuracy_state = getAccuracyLayerOutput(concat_output, model)
        for s in range(episods):
            mem_val = tf.config.experimental.get_memory_usage('GPU:0')
            print('######################### Print current time episods Starts  ############################################')
            # current date and time in Eastcost
            date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("date and time in East Coast:",date_time+ "\n")	
            print(str(s) + " - episode \n")
    
            if target == True:
                break
            total_reward = 0 
            state = derived_concat_output
            file_state_values.write(np.array2string(state, precision=8, separator=',') + "\n")
            prev_accuracy = (accuracy_state[0][0][0]).item()
            file_accuracy_values.write(str(prev_accuracy) + " - prev_accuracy\n")
            done = False
            steps_per_episode = 0
            # Training loop is simple i.e it interacts 
            # and stores experiences and learns at each action step.
            while not done: # cretae trejectories 
                # Agent can choose the action based upon the observations of the environment
                vars = 0
                action = agent.act(state)

                #action = agent.act(state, evaluate=True) when evaluate policy

                actionArray = tf.keras.backend.get_value(action)
                print('######################### actionArray ############################################')
                print(actionArray)
                file_actions.write(np.array2string(actionArray, precision=8, separator=',') + "\n")
                # Get new state, reward and done from the environment
                # Huber how does it take step...where is the execution.
                # How do we simulate in our case step here?
    
                next_state, next_accuracy, reward, done, steps_per_episode  = returnStepValues(action, alpha, state, prev_accuracy, steps_per_episode,max_steps_per_episode, model, done)   
                file_state_values.write(np.array2string(next_state, precision=8, separator=',') + " - next_state\n")
                file_accuracy_values.write(str(next_accuracy) + " - next_accuracy\n")
                print('######################### reward #########################################')
                print(reward)  
                file_reward.write(str(reward)+ "\n")         
                # Function to  next_state = state + alpha * action
                # reward  = diff between the accuracy of (next state and state)
                # done = when is when we acceed 20 steps
                # save this transition
                agent.savexp(state, next_state, actionArray, done, reward)
                
                # Agent has to learn now
                agent.train(file_target_actions,file_target_next_state_values,file_target_next_state_values2,
                            file_actor_loss,file_critic_value,file_critic_value2, file_critic_loss,file_critic_loss2,
                            file_new_policy_actions,file_next_state_target_value, file_target_values)
                
                # Save the current state of the environment to the new state
                state = next_state
                prev_accuracy = next_accuracy
                file_state_values.write(np.array2string(state, precision=8, separator=',') + " - state\n")
                file_accuracy_values.write(str(prev_accuracy) + " - prev_accuracy\n")

                #Total score will be added here
                total_reward += reward
                if done:
                    file_totalReward.write(str(total_reward) + "\n")
                    ep_reward.append(total_reward)
                    #calculate the average to get an idea if our agent is learning or not.
                    avg_reward = total_reward / max_steps_per_episode
                    avg_rewards_list.append(avg_reward)
                    total_avgr.append(avg_reward)
                    t6 = round(time.time() * 1000)
                    file_avgReward.write(str(avg_reward)+ " for episode - " +  str(s) + " at " + str(t6) + "\n")
            file_totalReward.flush()
            file_avgReward.flush()
            file_reward.flush()
            file_actions.flush()
            file_actor_loss.flush()   
            file_critic_value.flush() 
            file_critic_value2.flush()
            file_critic_loss.flush()  
            file_critic_loss2.flush()
            file_new_policy_actions.flush()
            file_state_values.flush() 
            file_accuracy_values.flush() 
            file_target_next_state_values2.flush() 
            file_next_state_target_value.flush()
            file_target_values.flush()

        # Save weights in a file
        agent.actor_main.save_weights(filepath + "/non_random/" + "actor_main_weights")
        agent.actor_target.save_weights(filepath + "/non_random/" + "actor_target_weights")
        agent.critic_main.save_weights(filepath + "/non_random/" + "critic_main_weights")
        agent.critic_main2.save_weights(filepath + "/non_random/" + "critic_main2_weights")
        agent.critic_target.save_weights(filepath + "/non_random/" + "critic_target_weights")
        agent.critic_target2.save_weights(filepath + "/non_random/" + "critic_target2_weights")


        file_totalReward.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_avgReward.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_reward.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_actions.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_actor_loss.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_critic_value.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_critic_value2.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_critic_loss.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_critic_loss2.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_new_policy_actions.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_state_values.write( "Episode - " +  str(s) + " Finsihed"  + "\n") 
        file_accuracy_values.write( "Episode - " +  str(s) + " Finsihed"  + "\n") 
        file_target_next_state_values2.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_next_state_target_value.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_target_values.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_totalReward.close()
        file_avgReward.close()
        file_reward.close()
        file_actions.close()
        file_actor_loss.close()   
        file_critic_value.close() 
        file_critic_value2.close()
        file_critic_loss.close()  
        file_critic_loss2.close()
        file_new_policy_actions.close()
        file_state_values.close() 
        file_accuracy_values.close() 
        file_target_next_state_values2.close()
        file_next_state_target_value.close()
        file_target_values.close()

# Now train actor critic
# The main program, Main loop 

## Phase II use random test network for every episode.
    #tf.config.experimental.reset_memory_stats('GPU:0')
    else:

        # load weights in a file
        #agent.actor_main.load_weights(filepath + "/random/" + "actor_main_weights")
        #agent.actor_target.load_weights(filepath + "/random/" + "actor_target_weights")
        #agent.critic_main.load_weights(filepath + "/random/" + "critic_main_weights")
        #agent.critic_main2.load_weights(filepath + "/random/" + "critic_main2_weights")
        #agent.critic_target.load_weights(filepath + "/random/" + "critic_target_weights")
        #agent.critic_target2.load_weights(filepath + "/random/" + "critic_target2_weights")



        file_totalReward_R = open('/home/bvadhera/huber/rl_results/totalReward_R.csv', 'w')
        file_avgReward_R = open('/home/bvadhera/huber/rl_results/avgReward_R.csv', 'w')
        file_Reward_R = open('/home/bvadhera/huber/rl_results/reward_R.csv', 'w')
        file_actions_R = open('/home/bvadhera/huber/rl_results/actions_R.csv', 'w')
        file_actor_loss_R = open('/home/bvadhera/huber/rl_results/actor_loss_R.csv', 'w')   
        file_critic_value_R = open('/home/bvadhera/huber/rl_results/critic_value_R.csv', 'w')  
        file_critic_value2_R = open('/home/bvadhera/huber/rl_results/critic_value2__R.csv', 'w') 
        file_critic_loss_R = open('/home/bvadhera/huber/rl_results/critic_loss_R.csv', 'w')  
        file_critic_loss2_R = open('/home/bvadhera/huber/rl_results/critic_loss2_R.csv', 'w')
        file_new_policy_actions_R  = open('/home/bvadhera/huber/rl_results/new_policy_actions_R.csv', 'w') 
        file_target_actions_R = open('/home/bvadhera/huber/rl_results/target_actions_R.csv', 'w')
        file_target_next_state_values_R = open('/home/bvadhera/huber/rl_results/target_next_state_values_R.csv', 'w')
        file_target_next_state_values2_R = open('/home/bvadhera/huber/rl_results/target_next_state_values2_R.csv', 'w')
        file_next_state_target_value_R = open('/home/bvadhera/huber/rl_results/next_state_target_value_R.csv', 'w')
        file_target_values_R = open('/home/bvadhera/huber/rl_results/target_values_R.csv', 'w')
        file_state_values_R = open('/home/bvadhera/huber/rl_results/state_values_R.csv', 'w')
        file_accuracy_values_R = open('/home/bvadhera/huber/rl_results/accuracy_values_R.csv', 'w')
    
        
        #episods = 100  
        #episods = 2000
        episods = 10000
        ep_reward = []
        total_avgr = []
        target = False
        max_steps_per_episode = 300 # trejectories
        alpha = 0.05
        test_nn = test_nn_rl
        concat_output = getConcatinateLayerOutput(test_nn, model)
        derived_concat_output =  concat_output[0][0]
        accuracy_state = getAccuracyLayerOutput(concat_output, model)
        for s in range(episods):
            mem_val = tf.config.experimental.get_memory_usage('GPU:0')
            print('######################### Print current time episods Starts  ############################################')
            # current date and time in Eastcost
            date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("date and time in East Coast:",date_time+ "\n")	
            print(str(s) + " - episode \n")
    
            if target == True:
                break
            total_reward = 0 
            state = derived_concat_output
            file_state_values_R.write(np.array2string(state, precision=8, separator=',') + "\n")
            prev_accuracy = (accuracy_state[0][0][0]).item()
            file_accuracy_values_R.write(str(prev_accuracy) + " - prev_accuracy\n")
            done = False
            steps_per_episode = 0
            # Training loop is simple i.e it interacts 
            # and stores experiences and learns at each action step.
            while not done: # cretae trejectories 
                # Agent can choose the action based upon the observations of the environment
                vars = 0
                action = agent.act(state)

                #action = agent.act(state, evaluate=True) when evaluate policy

                actionArray = tf.keras.backend.get_value(action)
                print('######################### actionArray ############################################')
                print(actionArray)
                file_actions_R.write(np.array2string(actionArray, precision=8, separator=',') + "\n")
                # Get new state, reward and done from the environment
                # Huber how does it take step...where is the execution.
                # How do we simulate in our case step here?
    
                next_state, next_accuracy, reward, done, steps_per_episode  = returnStepValues(action, alpha, state, prev_accuracy, steps_per_episode,max_steps_per_episode, model, done)   
    
                file_state_values_R.write(np.array2string(next_state, precision=8, separator=',') + " - next_state\n")
                file_accuracy_values_R.write(str(next_accuracy) + " - next_accuracy\n")
                print('######################### reward #########################################')
                print(reward)  
                file_Reward_R.write(str(reward)+ "\n")         
                # Function to  next_state = state + alpha * action
                # reward  = diff between the accuracy of (next state and state)
                # done = when is when we acceed 20 steps
                # save this transition
                agent.savexp(state, next_state, actionArray, done, reward)
                
                # Agent has to learn now
                agent.train(file_target_actions_R,file_target_next_state_values_R,file_target_next_state_values2_R,
                            file_actor_loss_R,file_critic_value_R,file_critic_value2_R, file_critic_loss_R,file_critic_loss2_R,
                            file_new_policy_actions_R,file_next_state_target_value_R, file_target_values_R)
                
                # Save the current state of the environment to the new state
                state = next_state
                prev_accuracy = next_accuracy
                file_state_values_R.write(np.array2string(state, precision=8, separator=',') + " - state\n")
                file_accuracy_values_R.write(str(prev_accuracy) + " - prev_accuracy\n")

                #Total score will be added here
                total_reward += reward
                if done:
                    file_totalReward_R.write(str(total_reward) + "\n")
                    ep_reward.append(total_reward)
                    #calculate the average to get an idea if our agent is learning or not.
                    avg_reward = total_reward / max_steps_per_episode 
                    avg_rewards_list.append(avg_reward)
                    total_avgr.append(avg_reward)
                    t6 = round(time.time() * 1000)
                    file_avgReward_R.write(str(avg_reward)+ " for episode - " +  str(s) + " at " + str(t6) + "\n")
            file_totalReward_R.flush()
            file_avgReward_R.flush()
            file_Reward_R.flush()
            file_actions_R.flush()
            file_actor_loss_R.flush()   
            file_critic_value_R.flush() 
            file_critic_value2_R.flush()
            file_critic_loss_R.flush()  
            file_critic_loss2_R.flush()
            file_new_policy_actions_R.flush()
            file_state_values_R.flush() 
            file_accuracy_values_R.flush() 
            file_target_next_state_values2_R.flush() 
            file_next_state_target_value_R.flush()
            file_target_values_R.flush()
            #Choose randon network for next episode
            num_layers,num_node1,num_node2 = getRandomLegalNetwork(divideBy,maxNumOfLayers, minNumOfLayers ) # generate valid normalize network
            test_nn = pd.DataFrame({'F1': [0], 'F2': [1],'F3': [0], 'F4': [0],'F5': [0],
                    'F6': [0],'F7': [0], 'F8': [0], 'F9': [0],'F10': [0], 'num_layers_N': [num_layers],
                    'num_node1_N': [num_node1],'num_node2_N': [num_node2]}, index=['50112'])
            print("test_nn_fabricated.head() ")
            print(test_nn.head())
            concat_output = getConcatinateLayerOutput(test_nn, model)
            derived_concat_output =  concat_output[0][0]
            accuracy_state = getAccuracyLayerOutput(concat_output, model)

        # Save weights in a file
        agent.actor_main.save_weights(filepath + "/random/" + "actor_main_weights")
        agent.actor_target.save_weights(filepath + "/random/" + "actor_target_weights")
        agent.critic_main.save_weights(filepath + "/random/" + "critic_main_weights")
        agent.critic_main2.save_weights(filepath + "/random/" + "critic_main2_weights")
        agent.critic_target.save_weights(filepath + "/random/" + "critic_target_weights")
        agent.critic_target2.save_weights(filepath + "/random/" + "critic_target2_weights")


        file_totalReward_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_avgReward_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_Reward_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_actions_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_actor_loss_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_critic_value_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_critic_value2_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_critic_loss_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_critic_loss2_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_new_policy_actions_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_state_values_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n") 
        file_accuracy_values_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n") 
        file_target_next_state_values2_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_next_state_target_value_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_target_values_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
        file_totalReward_R.close()
        file_avgReward_R.close()
        file_Reward_R.close()
        file_actions_R.close()
        file_actor_loss_R.close()   
        file_critic_value_R.close() 
        file_critic_value2_R.close()
        file_critic_loss_R.close()  
        file_critic_loss2_R.close()
        file_new_policy_actions_R.close()
        file_state_values_R.close() 
        file_accuracy_values_R.close() 
        file_target_next_state_values2_R.close()
        file_next_state_target_value_R.close()
        file_target_values_R.close()


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


#generateEmbeddingsCritic(grid_DataSet)
generateEmbeddingsActor(grid_DataSet)
#-------------------
random100 = True  # Use random NN of 100 NN to do policy evaluation
writePolicyOutPutFile = "/home/bvadhera/huber/38234EpisodsRandomRLRandomPolicyRandom100NN/policyOutPutFor.csv"
myPolicyOutPutFile = open(writePolicyOutPutFile, 'w')
##  TODO - CHange the headings as per the results
headerPolicyOutPutFile = ['encoded_x','encoded_y', 'layers', 'nnode1','nnode2','accuracy','reward','total_reward','critic_value','actor_value_1','actor_value_2'] 
writerPolicyOutPut = csv.DictWriter(myPolicyOutPutFile, fieldnames=headerPolicyOutPutFile) 
writerPolicyOutPut.writeheader()

if random100:
    #=========
    for index1 in range(100):
        sameStartNode = False
        encoded_x, encoded_y,n_layers,num_node1,num_node2, prev_accuracy,reward,total_reward,qValue,qActions_1, qActions_2 = getPolicyoutput(index1)
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
        state, num_layers,num_node1,num_node2, accuracy_state = getStateAndAccuracy(model,divideBy,maxNumOfLayers, minNumOfLayers)
else:
    #  same as testNN
    sameStartNode = True
    encoded_x, encoded_y,n_layers,num_node1,num_node2, prev_accuracy,reward,total_reward,qValue,qActions_1, qActions_2 = getPolicyoutput(index1=0)
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
'''