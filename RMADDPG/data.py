import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from gym.utils import seeding
from gym import spaces
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
#x_n = np.divide(x_n,200)

p = x_n[2][0:24 , 0]
l = x_n[2][0:24 , 1]

sns.set(style="darkgrid", font_scale=1.5)
sns.set_style('whitegrid',{'font.sans-serif':['simhei','Arial']}) 
plt.rcParams['font.sans-serif'] = ['SimHei']#用来显示中文标签
plt.rcParams['axes.unicode_minus'] = False#用来正常显示负号
plt.figure(figsize = (10,6))
#sns.tsplot(time=np.arange(len(x2)), data=x2, color="b", condition="奖励")
#sns.tsplot(time=np.arange(len(x1)), data=x1, color="r", condition="平均奖励")
sns.tsplot(time=np.arange(len(l)), data=l, marker='*', color="g", condition="Load")
sns.tsplot(time=np.arange(len(p)), data=p, marker='o',color="y", condition="PV")
#sns.tsplot(time=np.arange(len(x3)), data=x5, color="y", condition="action")
#plt.bar(range(0, 24 ),np.linspace(-200, 200, num = 11, endpoint = True)[x2[-24:]], 
 #       facecolor = "b", edgecolor = "k", label = "action")
plt.xlabel("Hour",labelpad=-1.8)
plt.ylabel("Power(KW)",labelpad=5)
plt.legend(loc='upper left')
 
#plt.savefig('fig8.svg',format='svg')
#print(x1[-24:],x2[-24:],x3[-24:])
plt.show() 