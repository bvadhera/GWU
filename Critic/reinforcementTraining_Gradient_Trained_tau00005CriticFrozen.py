# Reinforcement Training
from keras.layers import Input, Dense, Lambda, Concatenate
#import helperFunctions_JustAccuracyLamdaForwardLookupTreeActorCriticTrained
import helperFunctions_CriticFrozen
import randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from datetime import datetime
import math
import random
import os
import psutil
import sys
import gc
from csv import DictReader
import csv
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import numpy as np
import time
import pickle
import re
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import pandas as pd
from tensorflow import keras
import tensorflow as tf
print(tf.__version__)
import pickle
#from pympler.tracker import SummaryTracker


def rlTraining(test_nn_rl, accuracy_model, feature_Conversion_model,agent, divideBy, maxNumOfLayers, minNumOfLayers, current_dir):
    avg_rewards_list = []
    #filepath = "/home/bvadhera/huber/acc/"
    filepath = current_dir + "acc_Siamese/"
    test_DataSet = current_dir+"test_DataSet10.csv"
    # Now train actor critic
    # The main program, Main loop
    # Phase I use previos test network for every episode.
    with tf.device('GPU:0'):
        alpha = 0.005
        file_totalReward_R = open(
                    current_dir+'rl_results/totalReward_R_1.csv', 'a')
        file_avgReward_R = open(
            current_dir+'rl_results/avgReward_R_1.csv', 'a')
        file_Reward_R = open(current_dir+'rl_results/reward_R_1.csv', 'a')
        file_actions_R = open(current_dir+'rl_results/actions_R_1.csv', 'a')
        file_actor_loss_R = open(
            current_dir+'rl_results/actor_loss_R_1.csv', 'a')
        file_critic_value_R = open(
            current_dir+'rl_results/critic_value_R_1.csv', 'a')
        file_critic_value2_R = open(
            current_dir+'rl_results/critic_value2_R_1.csv', 'a')
        file_critic_loss_R = open(
            current_dir+'rl_results/critic_loss_R_1.csv', 'a')
        file_critic_loss2_R = open(
            current_dir+'rl_results/critic_loss2_R_1.csv', 'a')
        file_new_policy_actions_R = open(
            current_dir+'rl_results/new_policy_actions_R_1.csv', 'a')
        file_target_actions_R = open(
            current_dir+'rl_results/target_actions_R_1.csv', 'a')
        file_target_next_state_values_R = open(
            current_dir+'rl_results/target_next_state_values_R_1.csv', 'a')
        file_target_next_state_values2_R = open(
            current_dir+'rl_results/target_next_state_values2_R_1.csv', 'a')
        file_next_state_target_value_R = open(
            current_dir+'rl_results/next_state_target_value_R_1.csv', 'a')
        file_target_values_R = open(
            current_dir+'rl_results/target_values_R_1.csv', 'a')
        file_state_values_R = open(
            current_dir+'rl_results/state_values_R_1.csv', 'a')
        file_accuracy_values_R = open(
            current_dir+'rl_results/accuracy_values_R_1.csv', 'a')
        file_utility_values_R = open(
            current_dir+'rl_results/utility_values_R_1.csv', 'a')
        file_all_trained_states_R = open(current_dir+'critic_Siamese_stuff/trained_embeddings'  + '.csv', 'a')
        max_steps_per_episode = 300
        steps_per_episode = 0
        done = False
        
        #critic_DataSet = current_dir+"gridpoints_921600_DrivenReward_360-slice.csv"
        critic_DataSet = current_dir+"9NN_FeatureVectorTraining.csv"
        #critic_DataSet = test_DataSet

        # Open a csv reader called DictReader
        # iterate over each line as a ordered dictionary and print only few column by column name
        cdf = pd.read_csv(critic_DataSet)
        #cdf.columns = ['emb_x', 'emb_y', 'current_accuracy', 'legal']
         
        with open(current_dir+'rl_results/test_actions.csv', 'a') as actor_file:
            file_writer = csv.writer(actor_file)

            s = 0
            #while s in range(0,90000):  
            while s in range(0,360000):  # 120 episods
                i = 0
                #while s in range(0,6000000):  
                while i in range(0,20):
                    x = random.randint(0,899)
                    #x = 0
                    value = cdf.iloc[[x]]
                    x_value = list(value['emb_x'])[0]
                    y_value = list(value['emb_y'])[0]
                    #current_accuracy = list(value['Accuracy'])[0]
                    #current_legal = list(value['Legal'])[0]
                    xy_value = np.append(x_value, y_value)
                    FV1 = list(value['NoOfAttributes'])[0]
                    FV2 = list(value['NoOfClasses'])[0]
                    FV3 = list(value['DataSetSize'])[0]
                    FV4 = list(value['AttributeType'])[0]
                    FV5 = list(value['EntropyLabel'])[0]
                    FV6 = list(value['AvgEntrophyFeatures'])[0]
                    FV7 = list(value['AvgCorelationBetFeatures'])[0]
                    FV8 = list(value['2_6_8_TrainingAccuracy'])[0]
                    FV9 = list(value['2_6_8_TestAccuracy'])[0]
                    FV10 = list(value['1_5_0_TrainingAccuracy'])[0]
                    FV11 = list(value['1_5_0_TestAccuracy'])[0]
                    FV12 = list(value['2_6_20_TrainingAccuracy'])[0]
                    FV13 = list(value['2_6_20_TestAccuracy'])[0]

                    oneHotVector = np.array([[FV1,FV2,FV3,FV4,FV5,FV6,FV7,FV8,FV9,FV10,FV11,FV12,FV13]])
                    oneHotVector = tf.convert_to_tensor(oneHotVector, dtype=tf.float32)
                    encoder_row_Hot = feature_Conversion_model.predict(oneHotVector)  
                    #featureVectors = tf.keras.backend.get_value(encoder_row_Hot)
                    xy_value = np.append(x_value, y_value)
                    list_xy_value = []
                    list_xy_value.append(xy_value)
                    xy_valueT = tf.convert_to_tensor(list_xy_value, dtype=tf.float32)
                    state = xy_valueT
                    state_T = Concatenate()([state ,encoder_row_Hot])
                    state = tf.keras.backend.get_value(state_T)
                    accuracy = accuracy_model.predict(state)
                    current_accuracy = (accuracy[0][0]).item()
                    current_legal = (accuracy[0][1]).item()
                    # ToDo: collect tensor for 20 and call act to save time
                    action = agent.act(state[0], False)
                    actionArray = tf.keras.backend.get_value(action)
                    file_utility_values_R = open(current_dir+'rl_results/utility_values_R_1.csv', 'a')
                    
                    # # redoing the state as Accuracy network only takes vector of size 8 ## CHECK WITH DOCTOR HUBER
                    # FV11 = list(value['FV1'])[0]
                    # FV22 = list(value['FV2'])[0]
                    # FV33 = list(value['FV3'])[0]
                    # FV44 = list(value['FV4'])[0]
                    # FV55 = list(value['FV5'])[0]
                    # FV66 = list(value['FV6'])[0]
                    # state1 = np.append(xy_value, [FV11, FV22, FV33, FV44, FV55, FV66], axis=0)

                    # ToDo: send the tensor for 20 to speedup   
                    next_state,  next_accuracy, n_legal, reward, done, steps_per_episode, depth = helperFunctions_CriticFrozen.returnStepValues(
                                                                                actionArray, alpha, state[0],current_accuracy,current_legal, steps_per_episode, max_steps_per_episode,
                                                                            accuracy_model,done, agent,file_utility_values_R)
                    # save this transition
                    agent.savexp(state[0], next_state, actionArray, False, reward, next_accuracy, depth)
                    print("Done returnStepValues: ", str(i) +  "\n") 
                    i += 1
                
                '''
                < MAKE SURE WE DO RANDOM selection by savin the emb_x and emb_y and Accuracy into a buffer >
            
                1. 20 random points from emb dataset and apply the actor network with evaluation = False (actor.act()) then callable
                    next_state, next_accuracy, reward, done, steps_per_episode, depth = helperFunctions_JustAccuracyLamdaWithInputDataNextStep.returnStepValues(
                                                                            actionArray, alpha, state, row["prev_accuracy"], steps_per_episode, max_steps_per_episode,
                                                                            model,done, agent,file_utility_values_R)


                2. add these 20 next_state, next_accuracy, reward, done, steps_per_episode, depth  to the buffer 


                3. Once we have 300 points the training happens 

                4. repeat 1 - 3 again 

                5. Once the buffer grown beyond 40000 then start throwing the least used first 20 points  or sample from last 40000

                '''      
                print("===============================INDEX============================================")
                print("===============================INDEX============================================")
                print("===============================INDEX============================================")
                print("===============================INDEX============================================")
                print("===============================INDEX============================================")
                print("===============================INDEX============================================")
                print("Done episodes: ", str(s) +  "\n") 
                # Agent has to learn now
                completed = agent.train(file_target_actions_R, file_target_next_state_values_R, file_target_next_state_values2_R,
                            file_actor_loss_R, file_critic_value_R, file_critic_value2_R, file_critic_loss_R, file_critic_loss2_R,
                            file_new_policy_actions_R, file_next_state_target_value_R, file_target_values_R,
                            file_all_trained_states_R, agent,s)
                print("Done agent.train: ", str(s) +  "\n") 
                
                if completed == 1:
                    listOfActors = randomNetworkGeneration_UpdatedAccuracyModelLamdaForwardLookupTreeActorCriticTrained.generateEmbeddingsActorTest(test_DataSet, agent,current_dir)
                    # write listOfgridData_DrivenAcc in a csv file
                    item_length = len(listOfActors)
                    print (item_length)
                    for i in range(item_length):
                        file_writer.writerow([str(element) for element in listOfActors[i]])
                    actor_file.flush()
                
                if  ((s % 10) == 0): 
                    # Save weights in a file AFTER EVERY 10 EPISODS  As it erases everytime we save old weight
                    agent.actor_main.save_weights(
                        filepath + "test_siamese_FV_weights/" + "actor_main_weights")
                    agent.actor_target.save_weights(
                        filepath + "test_siamese_FV_weights/" + "actor_target_weights")
                    agent.critic_main.save_weights(
                        filepath + "test_siamese_FV_weights/" + "critic_main_weights")
                    agent.critic_main2.save_weights(
                        filepath + "test_siamese_FV_weights/" + "critic_main2_weights")
                    agent.critic_target.save_weights(
                        filepath + "test_siamese_FV_weights/" + "critic_target_weights")
                    agent.critic_target2.save_weights(
                        filepath + "test_siamese_FV_weights/" + "critic_target2_weights")
                
                s += 1

        if True : # Write statements to a file
                
                file_actions_R.write("Finsihed" + "\n")
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
                file_utility_values_R.close()
                file_all_trained_states_R.close()
                file_target_values_R.close()

    return agent
