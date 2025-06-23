# Reinforcement Training
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

import randomNetworkGeneration
import helperFunctions

def rlTraining(test_nn_rl, model, agent, divideBy,maxNumOfLayers, minNumOfLayers,current_dir):
    avg_rewards_list = []
    filepath = "/home/bvadhera/huber/acc/"
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
            file_totalReward = open(current_dir+'rl_results/totalReward.csv', 'w')
            file_avgReward = open(current_dir+'rl_results/avgReward.csv', 'w')
            file_reward = open(current_dir+'rl_results/reward.csv', 'w')
            file_actions = open(current_dir+'rl_results/actions.csv', 'w')
            file_actor_loss = open(current_dir+'rl_results/actor_loss.csv', 'w')   
            file_critic_value = open(current_dir+'rl_results/critic_value.csv', 'w')  
            file_critic_value2 = open(current_dir+'rl_results/critic_value2.csv', 'w') 
            file_critic_loss = open(current_dir+'rl_results/critic_loss.csv', 'w')  
            file_critic_loss2 = open(current_dir+'rl_results/critic_loss2.csv', 'w')
            file_new_policy_actions  = open(current_dir+'rl_results/new_policy_actions.csv', 'w') 
            file_target_actions = open(current_dir+'rl_results/target_actions.csv', 'w')
            file_target_next_state_values = open(current_dir+'rl_results/target_next_state_values.csv', 'w')
            file_target_next_state_values2 = open(current_dir+'rl_results/target_next_state_values2.csv', 'w')
            file_next_state_target_value = open(current_dir+'rl_results/next_state_target_value.csv', 'w')
            file_target_values = open(current_dir+'rl_results/target_values.csv', 'w')
            file_state_values = open(current_dir+'rl_results/state_values.csv', 'w')
            file_accuracy_values = open(current_dir+'rl_results/accuracy_values.csv', 'w')
            file_all_trained_states = open(current_dir+'rl_results/trained_embeddings.csv', 'w')
        
        
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
            concat_output = randomNetworkGeneration.getConcatinateLayerOutput(test_nn, model)
            derived_concat_output =  concat_output[0][0]
            accuracy_state = randomNetworkGeneration.getAccuracyLayerOutput(concat_output, model)
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
        
                    next_state, next_accuracy, reward, done, steps_per_episode  = helperFunctions.returnStepValues(action, alpha, state, prev_accuracy, steps_per_episode,max_steps_per_episode, model, done)   
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
                                file_new_policy_actions,file_next_state_target_value, file_target_values,file_all_trained_states,agent)
                    
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
                file_all_trained_states.flush()

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
            file_all_trained_states.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
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
            file_all_trained_states.close()

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



            file_totalReward_R = open(current_dir+'rl_results/totalReward_R.csv', 'w')
            file_avgReward_R = open(current_dir+'rl_results/avgReward_R.csv', 'w')
            file_Reward_R = open(current_dir+'rl_results/reward_R.csv', 'w')
            file_actions_R = open(current_dir+'rl_results/actions_R.csv', 'w')
            file_actor_loss_R = open(current_dir+'rl_results/actor_loss_R.csv', 'w')   
            file_critic_value_R = open(current_dir+'rl_results/critic_value_R.csv', 'w')  
            file_critic_value2_R = open(current_dir+'rl_results/critic_value2__R.csv', 'w') 
            file_critic_loss_R = open(current_dir+'rl_results/critic_loss_R.csv', 'w')  
            file_critic_loss2_R = open(current_dir+'rl_results/critic_loss2_R.csv', 'w')
            file_new_policy_actions_R  = open(current_dir+'rl_results/new_policy_actions_R.csv', 'w') 
            file_target_actions_R = open(current_dir+'rl_results/target_actions_R.csv', 'w')
            file_target_next_state_values_R = open(current_dir+'rl_results/target_next_state_values_R.csv', 'w')
            file_target_next_state_values2_R = open(current_dir+'rl_results/target_next_state_values2_R.csv', 'w')
            file_next_state_target_value_R = open(current_dir+'rl_results/next_state_target_value_R.csv', 'w')
            file_target_values_R = open(current_dir+'rl_results/target_values_R.csv', 'w')
            file_state_values_R = open(current_dir+'rl_results/state_values_R.csv', 'w')
            file_accuracy_values_R = open(current_dir+'rl_results/accuracy_values_R.csv', 'w')

        
            episods = 1000
            ep_reward = []
            total_avgr = []
            target = False
            max_steps_per_episode = 300 # trejectories
        
            alpha = 0.05
            test_nn = test_nn_rl
            concat_output = randomNetworkGeneration.getConcatinateLayerOutput(test_nn, model)
            derived_concat_output =  concat_output[0][0]
            accuracy_state = randomNetworkGeneration.getAccuracyLayerOutput(concat_output, model)
            randomBufferState = []
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
                file_all_trained_states_R = open(current_dir+'critic_stuff/trained_embeddings'+ str(s) + '.csv', 'w')
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
        
                    next_state, next_accuracy, reward, done, steps_per_episode  = helperFunctions.returnStepValues(action, alpha, state, prev_accuracy, steps_per_episode,max_steps_per_episode, model, done)   
                    if (steps_per_episode > 250) :
                        randomBufferState.append(next_state)
                        #remove that value from list now
                        #dump 1500 size then start dumping first oldest 50
                        #Maintain the size of randomBufferState to 1500 and pop first 50
                        if len(randomBufferState) > 1500:
                            nnToRemove = len(randomBufferState) - 1500
                            randomBufferState =  randomBufferState[nnToRemove:]
                            
                    file_state_values_R.write(np.array2string(next_state[0], precision=8, separator=',') + " - next_state\n")
                    file_accuracy_values_R.write(str(next_accuracy) + " - next_accuracy\n")
                    print('######################### reward #########################################')
                    print(reward)  
                    
                    # Calculate Critic_value
                    #action = agent.act(state, True)
                    # Call Critic
                    # ERROR HUBER
                    # convert states and new states, rewards and actions to tensor.
                    states = np.array([state])
                    actions = np.array([action])
                    states = tf.convert_to_tensor(states, dtype= tf.float32)
                    actions = tf.convert_to_tensor(actions, dtype= tf.float32)
                    critic_value = tf.squeeze(agent.critic_main(states, actions), 1)
                    critic_value = (tf.keras.backend.get_value(critic_value))[0]
                    file_Reward_R.write(str(reward)+','+ str(state[0]) +','+ str(state[1]) +','+ str(prev_accuracy) +','+ str(critic_value)+ "\n")         
                    
                    print(" ########################################################## ")
                    print(" ########################################################## ")
                    print(" ########################################################## ")
                    print(" ########################################################## ")
                    print(" ########################################################## ")
                    print(" ########################################################## ")
                    print(" ########################################################## ")
                    print(" ########################################################## ")

                    print("file_Reward_R.write done ")
                    
                    print(" ########################################################## ")
                    print(" ########################################################## ")
                    print(" ########################################################## ")
                    print(" ########################################################## ")
                    print(" ########################################################## ")
                    print(" ########################################################## ")
                    print(" ########################################################## ")
                    print(" ########################################################## ")
                    
                    
                    
                    # Function to  next_state = state + alpha * action
                    # reward  = diff between the accuracy of (next state and state)
                    # done = when is when we acceed 20 steps
                    # save this transition
                    agent.savexp(state, next_state, actionArray, done, reward)
                    
                    # Agent has to learn now
                    agent.train(file_target_actions_R,file_target_next_state_values_R,file_target_next_state_values2_R,
                                file_actor_loss_R,file_critic_value_R,file_critic_value2_R, file_critic_loss_R,file_critic_loss2_R,
                                file_new_policy_actions_R,file_next_state_target_value_R, file_target_values_R,
                                file_all_trained_states_R,agent)
                    
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
                file_all_trained_states_R.close()
                x_0_1 = random.randint(0,1000)
                lglStrProb = 0.25
                if (s < 10) or (x_0_1 < lglStrProb*1000 ) :   
                    #Choose randon network for next episode
                    num_layers,num_node1,num_node2 = randomNetworkGeneration.getRandomLegalNetwork(divideBy,maxNumOfLayers, minNumOfLayers ) # generate valid normalize network
                    test_nn = pd.DataFrame({'F1': [0], 'F2': [1],'F3': [0], 'F4': [0],'F5': [0],
                            'F6': [0],'F7': [0], 'F8': [0], 'F9': [0],'F10': [0], 'num_layers_N': [num_layers],
                            'num_node1_N': [num_node1],'num_node2_N': [num_node2]}, index=['50112'])
                    print("test_nn_fabricated.head() ")
                    print(test_nn.head())
                    concat_output = randomNetworkGeneration.getConcatinateLayerOutput(test_nn, model)
                    derived_concat_output =  concat_output[0][0]
                else:  # after 300 episodes take network from buffered states
                    concat_output = []   
                    start_point = 0
                    if len(randomBufferState) < 1500:
                        x = random.randint(start_point,len(randomBufferState))
                    else:
                        x = random.randint(start_point,1499)
                    print(x)
                    a = np.array([randomBufferState[x].tolist(),])
                    concat_output.append(a)
                    derived_concat_output =  concat_output[0][0]
                accuracy_state = randomNetworkGeneration.getAccuracyLayerOutput(concat_output, model)
                
                # HACK to save actor output after every 100 episodes
                '''
                grid_DataSet = current_dir+"gridpoints_160.csv"
                tstList = [0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,
                        2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,
                        4100,4200,4300,4400,4500,4600,4700,4800,4900,5000]
                if  s in tstList:
                    df_grid = pd.read_csv(grid_DataSet)
                    print(df_grid.head())
                    listOfgridData = df_grid.values.tolist()
                    listOfActorCritic = []
                    for i in listOfgridData:
                        concat_output = i
                        emb_x =  concat_output[0]
                        emb_y =  concat_output[1]
                        state = np.array(concat_output)
                        states = tf.convert_to_tensor([state], dtype=tf.float32)
                        action = agent.actor_main(states)
                        qActions_1 = (tf.keras.backend.get_value(action))[0][0]
                        qActions_2 = (tf.keras.backend.get_value(action))[0][0]
                        #actions = (tf.keras.backend.get_value(action))[0]
                        #actions = tf.convert_to_tensor(actions, dtype= tf.float32)
                        criticValue = tf.squeeze(agent.critic_main(states, action), 1)
                        criticValue = (tf.keras.backend.get_value(criticValue))[0]
                        NN_DrivenActorCritic = (emb_x,emb_y,qActions_1,qActions_2,criticValue)
                        listOfActorCritic.append(NN_DrivenActorCritic)
                    # write listOfgridData_DrivenAcc in a csv file
                    item_length = len(listOfActorCritic)
                    print (item_length)
                    with open(current_dir+'gridpoints_DrivenActorCritic' + str(s) + '.csv', 'w') as actor_file:
                        file_writer = csv.writer(actor_file)
                        for i in range(item_length):
                            file_writer.writerow([str(element) for element in listOfActorCritic[i]])
              
            
            # HACK
            # Save all embeddings from rbuffer
            all_trained_states = open(current_dir+'rl_results/allTrained_embeddings.csv', 'w')
            allStates = agent.memory.state_memory
            allStates_df = pd.DataFrame(allStates)
            allEmbeddings_df = allStates_df[[0, 1]]
            # 1000 episods * 301 each episodes
            #usefulBuff = 3010001
            # Slicing last n rows
            #usefullEmbeddings_df = allEmbeddings_df[:-usefulBuff]
            allEmbeddings_df.rename(columns={0: 'emb_x', 1: 'emb_y'}, inplace=True)
            #Hack adding critic value to the embeddings
            allEmbeddings_df = allEmbeddings_df.reset_index()  # make sure indexes pair with number of rows
            critic_list = []
            for index, row in allEmbeddings_df.iterrows():
                print(row[0], row[1])
                state = [0,0,0,1,0,0,0,0,0,0,0,0]
                state[0] = row[0] 
                state[1] = row[1]
                action = agent.act(state, True)
                states = np.array([state])
                actions = np.array([action])
                states = tf.convert_to_tensor(states, dtype= tf.float32)
                actions = tf.convert_to_tensor(actions, dtype= tf.float32)
                critic = tf.squeeze(agent.critic_main(states, actions), 1)
                critic = (tf.keras.backend.get_value(critic))[0]
                critic_list.append(critic)
            prop = list(allEmbeddings_df.columns.values)
            allEmbeddings_df['critic'] = critic_list  
            allEmbeddings_df.to_csv(all_trained_states,header=False)   
            all_trained_states.flush()
            all_trained_states.close()

            '''

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
            #file_all_trained_states_R.write( "Episode - " +  str(s) + " Finsihed"  + "\n")
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
            

    return agent