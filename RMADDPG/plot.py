import numpy as np                              # 导入numpy
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


file =  "total_reward_list_pomdp"+".pkl"
with open(os.path.join(file), "rb") as f:
    data = pickle.load(f)
x1 = data["mean"]
#x1 = x1[-24:]
file =  "total_reward_list_ddpg"+".pkl"
with open(os.path.join(file), "rb") as f:
    data = pickle.load(f)
x2 = data["mean"]

file =  "total_reward_list_D"+".pkl"
with open(os.path.join(file), "rb") as f:
    data = pickle.load(f)
x3 = data["mean"]

file =  "action_list_0"+".pkl"
with open(os.path.join(file), "rb") as f:
    data = pickle.load(f)
x6 = data["mean"]
x6=x6[-24:]
file =  "SOC_list_0"+".pkl"
with open(os.path.join(file), "rb") as f:
    data = pickle.load(f)
x7 = data["mean"]
x7=x7[-24:]
#x9 = np.sum([x4,x7],axis=0).tolist()
sns.set(style="darkgrid", font_scale=1.5)
sns.set_style('whitegrid',{'font.sans-serif':['simhei','Arial']}) 
plt.rcParams['font.sans-serif'] = ['SimHei']#用来显示中文标签
plt.rcParams['axes.unicode_minus'] = False#用来正常显示负号
sns.tsplot(time=np.arange(len(x3)), data=x3, color="b", condition="RMADDPG")
#sns.tsplot(time=np.arange(len(x1)), data=x1, color="r", condition="maddpg")
sns.tsplot(time=np.arange(len(x2)), data=x2, color="g", condition="DRPG")
#sns.tsplot(time=np.arange(len(x1)), data=x1, color="r", condition="SOC(%)")
sns.tsplot(time=np.arange(len(x2)), data=x1, color="y", condition="VFA")
#plt.bar(range(0, 24 ),x6[-24:0], 
  #      facecolor = "b", edgecolor = "k", label = "action")
plt.xlabel("Hour",labelpad=-1.8)
plt.ylabel("Power(KW)",labelpad=-2)
plt.legend(loc='lower right')
#plt.savefig('fig4_0.svg',format='svg')
plt.show()  
#print(x3,x6)