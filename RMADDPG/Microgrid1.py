import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from gym.utils import seeding
from gym import spaces
import os
import pickle
df_raw_1 = pd.read_csv("load_1.csv")
df_1 = df_raw_1.copy()
df_raw_2 = pd.read_csv("load_2.csv")
df_2 = df_raw_2.copy()
df_raw_3 = pd.read_csv("load_3.csv")
df_3 = df_raw_3.copy()
df_raw_4 = pd.read_csv("PV_1.csv")
df_4 = df_raw_4.copy()
df_raw_5 = pd.read_csv("PV_2.csv")
df_5 = df_raw_5.copy()
df_raw_6 = pd.read_csv("PV_3.csv")
df_6 = df_raw_6.copy()

PV_1 = df_4.iloc[:, 0:1].values
PV_2 = df_5.iloc[:, 0:1].values
PV_3 = df_6.iloc[:, 0:1].values
PV_n = [PV_1, PV_2, PV_3]

load_1 = df_1.iloc[:, 0:1].values
load_2 = df_2.iloc[:, 0:1].values
#load_2 = np.divide(load_2, 2)
load_3 = df_3.iloc[:, 0:1].values
load_n = [load_1, load_2, load_3]
load_1 = np.multiply(load_1,8)
load_1[14]=200
load_1[15]=100
load_1[16]=80
load_1[17]=50
load_1[18]=30
load_1[19]=20
load_1[20]=10
load_1[21]=5
load_1[22]=2
load_1[23]=1
load_3 = np.multiply(load_3,3)
x_1 = np.concatenate([PV_1,load_1], axis = -1)
x_2 = np.concatenate([PV_2,load_2], axis = -1)
x_3 = np.concatenate([PV_3,load_3], axis = -1)
x_n = [x_1, x_2, x_3]
x_n = np.divide(x_n,200)
class Microgrid(object):
    def __init__(self) -> None:
        self.action_space = [spaces.Box(low=-1, high=+1, shape=(2,), dtype=np.float32),spaces.Box(low=-1, high=+1, shape=(2,), dtype=np.float32),spaces.Box(low=-1, high=+1, shape=(2,), dtype=np.float32)]
        self.observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(2,), dtype=np.float32),spaces.Box(low=-np.inf, high=+np.inf, shape=(2,), dtype=np.float32),spaces.Box(low=-np.inf, high=+np.inf, shape=(2,), dtype=np.float32)]
        self.agent_num = 3
        self.n = 3
    def reset(self):
        self.day = 0
        self.hour = 0
        self.sub_agent_obs = []
        self.R = []
        SOC = np.array([0.1])
        for i in range(self.agent_num):
           sub_obs = np.concatenate((x_n[i][self.hour + self.day * 24 , :],SOC), axis = -1)
           self.sub_agent_obs.append(sub_obs)
        return self.sub_agent_obs
    
    def step(self,action_n):
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        sub_agent_info = {'n': [{},{},{}]}
        A = []    
        next_SOC_n = [np.array([0]), np.array([0]), np.array([0])]
        next_SOC_n_ = [np.array([0]), np.array([0]), np.array([0])]
        action_n = action_n[0]
        action_n[0][1] = np.clip(action_n[0][1],-0.5,0.5)
        action_n[1][1] = np.clip(action_n[1][1],-0.5,0.5)
        action_n[2][1] = np.clip(action_n[2][1],-0.5,0.5)
        for i in range (self.agent_num): 
            action_n[i][0] = np.clip(action_n[i][0],-1,1) 

            delta_SOC = np.array([action_n[i][0]/5])
            if action_n[i][0] <= 0:     
                next_SOC_n[i] = np.maximum(0.05, self.sub_agent_obs[i][2] + delta_SOC)
            else :
                next_SOC_n[i] = np.minimum(0.9, self.sub_agent_obs[i][2] + delta_SOC)

            if (self.sub_agent_obs[i][1] - self.sub_agent_obs[i][0] + (next_SOC_n[i] - self.sub_agent_obs[i][2])*5)[0] < 0:
                penalty2 = - (self.sub_agent_obs[i][1] - self.sub_agent_obs[i][0] + (next_SOC_n[i] - self.sub_agent_obs[i][2])*5)[0]
            else :
                penalty2 = 0  

            if i == 0:
                if action_n[i][1]-action_n[2][1] <= 0:
                    next_SOC_n_[i] = np.maximum(0.05, next_SOC_n[i] + np.array([(action_n[i][1]-action_n[2][1])/5]))
                else:
                    next_SOC_n_[i] = np.minimum(0.9, next_SOC_n[i] + np.array([(action_n[i][1]-action_n[2][1])/5]))
            elif i == 1:
                if action_n[i][1]-action_n[0][1] <= 0:  
                    next_SOC_n_[i] = np.maximum(0.05, next_SOC_n[i] + np.array([(action_n[i][1]-action_n[0][1])/5]))
                else:
                    next_SOC_n_[i] = np.minimum(0.9, next_SOC_n[i] + np.array([(action_n[i][1]-action_n[0][1])/5]))
            elif i == 2:
                if action_n[i][1]-action_n[1][1] <= 0:
                    next_SOC_n_[i] = np.maximum(0.05, next_SOC_n[i] + np.array([(action_n[i][1]-action_n[1][1])/5]))
                else:
                    next_SOC_n_[i] = np.minimum(0.9, next_SOC_n[i] + np.array([(action_n[i][1]-action_n[1][1])/5]))
                  
            sub_agent_reward.append(- (self.sub_agent_obs[i][1] - self.sub_agent_obs[i][0] + (next_SOC_n_[i] - self.sub_agent_obs[i][2])*5)[0] - penalty2)             

            sub_agent_done.append(False)
            self.sub_agent_obs[i] = np.concatenate((x_n[i][self.hour+1 + self.day * 24 , :],next_SOC_n_[i]), axis = -1)
        self.R.append(sub_agent_reward)
        if self.hour == 0:
           x = [0,0,0]
        else:
           x = self.R[-2:][0]
        for i in range(3):
           A.append(self.R[-1:][0][i]-x[i])
        sub_agent_reward = [[sub_agent_reward[0],A[0]],[sub_agent_reward[1],A[1]],[sub_agent_reward[2],A[2]]]
        if self.hour < 23 :
          self.hour += 1
        else :
          self.day += 1
          self.hour = 0    
   
        return self.sub_agent_obs, sub_agent_reward, sub_agent_done, next_SOC_n

    def seed(self, seed=None):   #seed设置为任意整数后，随机值固定，如果设置随机值固定
      self.np_random, seed = seeding.np_random(seed)
      return [seed]