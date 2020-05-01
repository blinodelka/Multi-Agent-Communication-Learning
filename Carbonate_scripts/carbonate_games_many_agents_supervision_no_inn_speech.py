#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 13:30:17 2020

@author: marinadubova
"""

import numpy as np
#from PIL import Image
#import cv2
import matplotlib.pyplot as plt
import sys
if sys.version[0] == '3':
    import pickle
else:
    import cPickle as pickle
from matplotlib import style
import time
import collections
import random
import math
import csv

import keras 

from keras.layers import Lambda, Input, Dense, LSTM, SimpleRNN, GRU
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy, categorical_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.regularizers import l1, l2
from keras.utils import np_utils, plot_model
from keras.layers import concatenate
from keras.layers.core import Reshape
from sklearn.preprocessing import OneHotEncoder

#import pydot
import tensorflow as tf
import os

from keras import backend as K
#K.tensorflow_backend._get_available_gpus()

import sys

trial_index = int(sys.argv[1])
#trial_index = 5
print("trial index:" + str(trial_index))
class Two_Roles_Game_Many_Agents():

    def __init__(
        self, num_choices, winning_reward, mean_sample, talking_action_size, num_agents, mean_punishment_weight_talk, punishment_weight):
        self.num_choices = num_choices
        self.talking_action_size = talking_action_size
        self.winning_reward = winning_reward
        self.mean_sample = mean_sample
        previous_actions = []
        previous_talks = []
        self.mean_punishment_weight_talk = mean_punishment_weight_talk
        for i in range(num_agents):
            previous_actions.append(collections.deque(maxlen=mean_sample)) # for mean calculations
        for i in range(num_agents):
            previous_talks.append(collections.deque(maxlen=mean_sample))
        self.previous_talks = previous_talks
        self.previous_actions = previous_actions
        self.mean_punishment_weight = punishment_weight
        #self.who_talks = int(random.random()>0.5) # needs to be updated after each game
                
    def step(self, ag1_action, ag2_action, choose, sample, reward1 = 0, reward2 = 0): #choose true or false
        #talker_input = np.zeros((self.talking_action_size+self.talking_action_size+1,)) # no noise
        #talker_input[1:self.talking_action_size+1] = np.random.random((self.talking_action_size,)) #UNIFORM/bimodal NOISE + signal for the game role
        talker_input = np.random.random((self.talking_action_size+self.num_choices+1,)) #UNIFORM/bimodal NOISE + signal for the game role
        talker_input[0] = 1 # indicates that needs to talk
        #hearer_input = np.zeros((self.talking_action_size+self.talking_action_size+1,)) # no noise
        hearer_input = np.zeros((self.talking_action_size+self.num_choices+1,))
        # ADD DISTR OF PREVIOUS ACTIONS TO INPUT
        for i in range(self.num_choices):
            talker_input[self.talking_action_size+1+i] = self.previous_actions[sample[0]].count(i)/self.mean_sample
            hearer_input[self.talking_action_size+1+i] = self.previous_actions[sample[1]].count(i)/self.mean_sample
        
        if choose: # step of choice
            self.previous_actions[sample[0]].append(ag1_action)
            self.previous_actions[sample[1]].append(ag2_action)
            a1_rew = 0
            a2_rew = 0
            
            if len(self.previous_actions[sample[0]]) >= 2 and len(self.previous_actions[sample[1]]) >= 2:
                #print(self.previous_actions[sample[0]])
                #print(self.previous_actions[sample[0]].count(ag1_action))
                a1_rew = ((float(1)/self.num_choices) - float(self.previous_actions[sample[0]].count(ag1_action))/len(self.previous_actions[sample[0]])) * self.mean_punishment_weight
                a2_rew = ((float(1)/self.num_choices) - float(self.previous_actions[sample[1]].count(ag2_action))/len(self.previous_actions[sample[1]])) * self.mean_punishment_weight
                if a1_rew>0:
                    a1_rew = 0
                if a2_rew>0:
                    a2_rew = 0
                #print("a1:" + str(a1_rew) + "a2:" + str(a2_rew))

            if ag2_action == ag1_action: 
                reward1 = self.winning_reward + a1_rew
                reward2 = self.winning_reward + a2_rew
                
            else:
                reward1 = a1_rew #-1
                reward2 = a2_rew #-1
                                    
        else:       
            talker_action = ag1_action # first agent talks all the time
            self.previous_talks[sample[0]].append(ag1_action)
            hearer_input[talker_action+1] = 1
            if len(self.previous_talks[sample[0]]) >= 2:
                reward1 = ((float(1)/self.talking_action_size) - float(self.previous_talks[sample[0]].count(ag1_action))/len(self.previous_talks[sample[0]])) * self.mean_punishment_weight_talk
                if reward1 > 0:
                    reward1 = 0
        
        return(talker_input, hearer_input, reward1, reward2)


class DQNAgent_student_teacher:
    def __init__(self, talking_action_size, choices, beta, memory_size = 11): 
        self.memory = collections.deque(maxlen=memory_size)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.beta_1 = beta
        self.learning_rate_bad = 0.0001
        self.learning_rate_good = 0.0001
        self.talking_action_size = talking_action_size
        self.choices = choices
        self.model = self._build_model_student_teacher()
 
 
    def _build_model_student_teacher(self):
        # Neural Net for Deep-Q learning Model
        
        model = Sequential()
        #model.add(Dense(25, input_dim=self.talking_action_size + self.talking_action_size +self.talking_action_size+ 1, activation='relu')) # one input - the game role
        model.add(Dense(15, input_dim=self.talking_action_size + self.choices + self.talking_action_size+ 1, activation='relu')) # one input - the game role
        #model.add(Dense(15, activation='relu'))
        model.add(Dense(25, activation='relu'))
 
        model.add(Dense(self.choices, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate_bad, beta_1=self.beta_1, beta_2=0.99))
        return model
       
   
    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))
                           
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return [random.randrange(self.talking_action_size), random.randrange(self.choices)]
 
        
        max_q = -float("infinity")
        max_talk = None
        max_action = None
 
        for i in range(self.talking_action_size):
            talking = np.zeros(self.talking_action_size)
            talking[i] = 1
 
            state_cur = np.expand_dims(np.hstack([state, talking]), axis=0)
            act_values = self.model.predict(state_cur, batch_size=1)
 
            cur_max = np.max(act_values)
           
            if cur_max > max_q:
 
              max_q = cur_max
              max_action = np.argmax(act_values[0])
              max_talk = i
 
            assert max_talk is not None
            assert max_action is not None
 
        return [max_talk, max_action]
                           
    def replay(self, batch_size):
        for i in self.memory:
            minibatch = random.sample(self.memory, batch_size)
        my_x = []
        my_y = []
        for state, action, reward, next_state in minibatch:
            next_state = np.expand_dims(next_state, axis=0)
            #state = np.expand_dims(state, axis=0)
            target = reward
            one_hot_talk = np.zeros(self.talking_action_size)
            one_hot_talk[action[0]] = 1
            cur_state = np.expand_dims(np.hstack([state, one_hot_talk]), axis=0)
            target_f = self.model.predict(cur_state, batch_size=1)
            target_f[0][action[1]] = target # choose
            cur_state = np.squeeze(cur_state)
            target_f = np.squeeze(target_f)
            my_x.append(cur_state)
            my_y.append(target_f)
        
        my_x = np.array(my_x)
        my_y = np.array(my_y)
        self.model.train_on_batch(my_x, my_y)
 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def play_many_games_semisupervised(num_agents, num_episodes, inner_speech, learning_rate, punishment_weight, punishment_talk, agents_memory, replay, beta_1, p_peek):
    num_choices = 4
    num_talking_symbols = 4
    winning_reward = 1
    mean_sample = 100
    punishment_weight = punishment_weight
    num_agents = num_agents
    
    env = Two_Roles_Game_Many_Agents(num_choices, winning_reward, mean_sample, num_talking_symbols, num_agents, punishment_talk, punishment_weight) 

    agents = []
    self_talks = [[],[]]
    talks = [[],[]]
    self_acts = [[[],[]],[[],[]]]
    acts = [[],[]]
    self_scores = [[],[]]
    scores = [[],[]]
    samples = []
    joint_scores = [[],[]]
    
    for i in range(num_agents):
        x = DQNAgent_student_teacher(num_talking_symbols, num_choices, beta_1, agents_memory)
        x.learning_rate_bad = learning_rate
        agents.append(x)
        talks.append([])
        acts.append([])
        scores.append([])
        
        # Iterate the game
    episodes = num_episodes

    for e in range(episodes):
        if inner_speech:
            my_sample = list(np.random.choice(num_agents, 2, replace=True)) #random.sample(range(num_agents), k=2)#random.choices(range(num_agents), k=2) # with replacement
        else:    
            my_sample = random.sample(range(num_agents), 2) #no replacement - no self-talks
        #if my_sample[0] == my_sample[1]: # make inner speech less frequent
        #    if random.random() < 0.3:
        #        my_sample = random.choices(range(num_agents), k=2) 
        agent1 = agents[my_sample[0]] # agent1 always talks
        agent2 = agents[my_sample[1]]
        score1 = 0
        score2 = 0

        state1, state2, _, _ = env.step(0, 0, 0, my_sample) # 4 - doing nothing 

        # agent 1 talks
        action1 = agent1.act(state1)

        state1, state2, reward1, reward2 = env.step(action1[0], 0, 0, my_sample)     
        score1 += reward1

        action2 = agent2.act(state2)

        # everyone chooses
        next_state1, next_state2, reward1, reward2 = env.step(action1[1], action2[1], 1, my_sample)     

        score1 += reward1
        score2 += reward2

        agents[my_sample[1]].remember(state2, action2, reward2, next_state2)
        
        if random.random() < p_peek: # peek another's action 20% of times: mimiking 
            agents[my_sample[1]].remember(state2, action1, winning_reward, next_state2)
            
        agents[my_sample[0]].remember(state1, action1, reward1, next_state1)
        if random.random() < p_peek: 
            agents[my_sample[0]].remember(state1, action2, winning_reward, next_state1) # potential problem: without mean_action discount
            
        #if e %100 == 0:
        #    print("episode: {}/{}, score1: {}, score2: {}"
        #                  .format(e, episodes, score1, score2))

        if len(agent1.memory) >= replay and len(agent2.memory) >= replay: 
            agents[my_sample[0]].replay(replay)
            agents[my_sample[1]].replay(replay)
        else:
            print("replay more than memory")
		
        if my_sample[0] == my_sample[1]: #inner_speech
            self_talks[my_sample[0]].append(action1[0])
            self_acts[my_sample[0]][0].append(action1[1])
            self_acts[my_sample[0]][1].append(action2[1])
            self_scores[my_sample[0]].append(score1)
            self_scores[my_sample[0]].append(score2)
        else:
            talks[my_sample[0]].append(action1[0])
            talks[my_sample[1]].append(-1) #didn't talk
            acts[my_sample[0]].append(action1[1])
            acts[my_sample[1]].append(action2[1])
            scores[my_sample[0]].append(score1)
            scores[my_sample[1]].append(score2)
        joint_scores[my_sample[0]].append(score1)
        joint_scores[my_sample[1]].append(score2)
        samples.append(my_sample)
        
    return [talks, acts, scores, self_talks, self_acts, self_scores, joint_scores, samples]

cond_list = []
#lrates = [0.00001,0.0001,0.001,0.01,0.1]
punishment_weights = [3]
beta = [0.3]
memory_replay = [[10],[10]]
inner_speech_conditions = [0,1]
num_agents = [2,4,6,8,10,12]
supervision = [0.0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7, 0.8, 0.9]
samples = 10
#for lr in lrates:
#for inn in inner_speech_conditions:
for superv in supervision:
    for n in num_agents:
        for j in range(samples):
            cond_list.append([superv, n])
            
supervision_rate = cond_list[trial_index][0]
n = cond_list[trial_index][1]
d = play_many_games_semisupervised(n, 60000, 0, 0.0001, 4, 0, 10, 10, 0.3, supervision_rate)
d.append(cond_list[trial_index])

#plt.plot(np.convolve(d[2][0], np.ones((100,))/100, mode='valid'))
#plt.plot(np.convolve(d[2][1], np.ones((100,))/100, mode='valid'))

with open("lang_games/game{}.pkl".format(trial_index), "wb") as fp:   #Pickling
    pickle.dump(d, fp, protocol=pickle.HIGHEST_PROTOCOL)
