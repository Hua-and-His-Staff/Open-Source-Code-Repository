
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace as debug

def Grid_initializer(module):
    """ Parameter initializer for Atari models
    Initializes Linear, Conv2d, and LSTM weights.
    """
    classname = module.__class__.__name__

    if classname == 'Linear':
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2.))
        module.bias.data.zero_()
        
    elif classname == 'Conv2d':
        nn.init.orthogonal_(module.weight, scale=np.sqrt(2.))
        module.bias.data.zero_()
    

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, num_observations, num_actions, init_w=3e-3):
        super(Actor, self).__init__()
        self.DG = nn.Sequential(nn.Linear(num_observations,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),                                
                                  nn.Tanh(),
                                  nn.Linear(64,1),
                                  nn.Sigmoid())
        self.EV0 = nn.Sequential(nn.Linear(2,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),
                                  nn.Linear(64,1),
                                  nn.Sigmoid())  
        self.EV1 = nn.Sequential(nn.Linear(2,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),
                                  nn.Linear(64,1),
                                  nn.Sigmoid()) 
        self.EV2 = nn.Sequential(nn.Linear(2,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),
                                  nn.Linear(64,1),
                                  nn.Sigmoid())  
        self.EV3 = nn.Sequential(nn.Linear(2,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),
                                  nn.Linear(64,1),
                                  nn.Sigmoid()) 
        self.EV4 = nn.Sequential(nn.Linear(2,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),
                                  nn.Linear(64,1),
                                  nn.Sigmoid()) 
        
        self.AC0 = nn.Sequential(nn.Linear(3,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),
                                  nn.Linear(64,1),
                                  nn.Sigmoid())
        self.AC1 = nn.Sequential(nn.Linear(3,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),
                                  nn.Linear(64,1),
                                  nn.Sigmoid())
        self.AC2 = nn.Sequential(nn.Linear(3,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),
                                  nn.Linear(64,1),
                                  nn.Sigmoid())
        self.AC3 = nn.Sequential(nn.Linear(3,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),
                                  nn.Linear(64,1),
                                  nn.Sigmoid())
        self.AC4 = nn.Sequential(nn.Linear(3,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),
                                  nn.Linear(64,1),
                                  nn.Sigmoid())
        self.AC5 = nn.Sequential(nn.Linear(3,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),
                                  nn.Linear(64,1),
                                  nn.Sigmoid())
        self.AC6 = nn.Sequential(nn.Linear(3,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),
                                  nn.Linear(64,1),
                                  nn.Sigmoid())
        self.AC7 = nn.Sequential(nn.Linear(3,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),
                                  nn.Linear(64,1),
                                  nn.Sigmoid())
        self.AC8 = nn.Sequential(nn.Linear(3,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),
                                  nn.Linear(64,1),
                                  nn.Sigmoid())
        self.AC9 = nn.Sequential(nn.Linear(3,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),
                                  nn.Linear(64,1),
                                  nn.Sigmoid())
        self.apply(Grid_initializer)
    
    #def init_weights(self, init_w):
    #    self.input.weight.data = fanin_init(self.input.weight.data.size())
        #self.hidden1.weight.data = fanin_init(self.hidden1.weight.data.size())
        #self.hidden2.weight.data = fanin_init(self.hidden2.weight.data.size())
    #    self.output.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        uDG = self.DG(x)
        s = torch.stack([x[:,0],x[:,5]],dim=1)
        uEV0 = self.EV0(s)
        s = torch.stack([x[:,0],x[:,6]],1)
        uEV1 = self.EV1(s)
        s = torch.stack([x[:,0],x[:,7]],1)
        uEV2 = self.EV2(s)
        s = torch.stack([x[:,0],x[:,8]],1)
        uEV3 = self.EV3(s)
        s = torch.stack([x[:,0],x[:,9]],1)
        uEV4 = self.EV4(s)
        
        s = torch.stack([x[:,0],x[:,4],x[:,10]],1)
        uAC0 = self.AC0(s)
        s = torch.stack([x[:,0],x[:,4],x[:,11]],1)
        uAC1 = self.AC1(s)
        s = torch.stack([x[:,0],x[:,4],x[:,12]],1)
        uAC2 = self.AC2(s)
        s = torch.stack([x[:,0],x[:,4],x[:,13]],1)
        uAC3 = self.AC3(s)
        s = torch.stack([x[:,0],x[:,4],x[:,14]],1)
        uAC4 = self.AC4(s)
        s = torch.stack([x[:,0],x[:,4],x[:,15]],1)
        uAC5 = self.AC5(s)
        s = torch.stack([x[:,0],x[:,4],x[:,16]],1)
        uAC6 = self.AC6(s)
        s = torch.stack([x[:,0],x[:,4],x[:,17]],1)
        uAC7 = self.AC7(s)
        s = torch.stack([x[:,0],x[:,4],x[:,18]],1)
        uAC8 = self.AC8(s)
        s = torch.stack([x[:,0],x[:,4],x[:,19]],1)
        uAC9 = self.AC9(s)
        return torch.cat([uDG,uEV0,uEV1,uEV2,uEV3,uEV4,uAC0,uAC1,uAC2,uAC3,uAC4,uAC5,uAC6,uAC7,uAC8,uAC9],1)
        
class Critic(nn.Module):
    def __init__(self, num_observations, num_actions, num_values, init_w=3e-3):
        super(Critic, self).__init__()

        
        #all
        self.Q_gen_bes = nn.Sequential(nn.Linear(num_observations + num_actions,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),
                                  nn.Linear(64, 64),
                                  nn.Tanh(),                               
                                  nn.Linear(64, 2))
        #state[:,0] state[:,5:10] action[:,1:6]
        self.Q_EV0 = nn.Sequential(nn.Linear(3,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),                               
                                  nn.Linear(64, 1))
        self.Q_EV1 = nn.Sequential(nn.Linear(3,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),                               
                                  nn.Linear(64, 1))
        self.Q_EV2 = nn.Sequential(nn.Linear(3,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),                               
                                  nn.Linear(64, 1))
        self.Q_EV3 = nn.Sequential(nn.Linear(3,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),                               
                                  nn.Linear(64, 1))
        self.Q_EV4 = nn.Sequential(nn.Linear(3,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),                               
                                  nn.Linear(64, 1))

        
        #state[:,0] state[:,4] state[:,10:20] action[:,6:16]
        self.Q_AC0 = nn.Sequential(nn.Linear(4,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),                              
                                  nn.Linear(64, 1))
        self.Q_AC1 = nn.Sequential(nn.Linear(4,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),                              
                                  nn.Linear(64, 1))    
        self.Q_AC2 = nn.Sequential(nn.Linear(4,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),                              
                                  nn.Linear(64, 1))  
        self.Q_AC3 = nn.Sequential(nn.Linear(4,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),                              
                                  nn.Linear(64, 1))    
        self.Q_AC4 = nn.Sequential(nn.Linear(4,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),                              
                                  nn.Linear(64, 1)) 
        self.Q_AC5 = nn.Sequential(nn.Linear(4,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),                              
                                  nn.Linear(64, 1))    
        self.Q_AC6 = nn.Sequential(nn.Linear(4,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),                              
                                  nn.Linear(64, 1))  
        self.Q_AC7 = nn.Sequential(nn.Linear(4,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),                              
                                  nn.Linear(64, 1))    
        self.Q_AC8 = nn.Sequential(nn.Linear(4,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),                              
                                  nn.Linear(64, 1)) 
        self.Q_AC7 = nn.Sequential(nn.Linear(4,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),                              
                                  nn.Linear(64, 1))    
        self.Q_AC9 = nn.Sequential(nn.Linear(4,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),                              
                                  nn.Linear(64, 1)) 
 
        self.apply(Grid_initializer)
    
    #def init_weights(self, init_w):
    #    self.input1.weight.data = fanin_init(self.input1.weight.data.size())
    #    self.input2.weight.data = fanin_init(self.input2.weight.data.size())
    #    self.hidden1.weight.data = fanin_init(self.hidden1.weight.data.size())     
    #    self.output.weight.data.uniform_(-init_w, init_w)
        #state[0]: t
        #state[1]: pv-laod
        #state[2]: PG
        #state[3]: soc
        #state[4]: Tout
        #state[5:10]: E_EV
        #state[10:20]: T_EV
        #action[:,0]: u^DG
        #action[:,1:6]: u^EV
        #action[:,6:16]: u^AC
    def forward(self, xs):
        x, a = xs
        xa = torch.cat([x,a],1)
        Q_g_b = self.Q_gen_bes(xa)
        Q_EV0 = self.Q_EV0(torch.stack((x[:,0],x[:,5],a[:,1]),1))
        Q_EV1 = self.Q_EV1(torch.stack((x[:,0],x[:,6],a[:,2]),1))
        Q_EV2 = self.Q_EV2(torch.stack((x[:,0],x[:,7],a[:,3]),1))
        Q_EV3 = self.Q_EV3(torch.stack((x[:,0],x[:,8],a[:,4]),1))
        Q_EV4 = self.Q_EV4(torch.stack((x[:,0],x[:,9],a[:,5]),1))
        Q_AC0 = self.Q_AC0(torch.stack((x[:,0],x[:,4],x[:,10],a[:,6]),1))
        Q_AC1 = self.Q_AC1(torch.stack((x[:,0],x[:,4],x[:,11],a[:,7]),1))
        Q_AC2 = self.Q_AC2(torch.stack((x[:,0],x[:,4],x[:,12],a[:,8]),1))
        Q_AC3 = self.Q_AC3(torch.stack((x[:,0],x[:,4],x[:,13],a[:,9]),1))
        Q_AC4 = self.Q_AC4(torch.stack((x[:,0],x[:,4],x[:,14],a[:,10]),1))
        Q_AC5 = self.Q_AC5(torch.stack((x[:,0],x[:,4],x[:,15],a[:,11]),1))
        Q_AC6 = self.Q_AC6(torch.stack((x[:,0],x[:,4],x[:,16],a[:,12]),1))
        Q_AC7 = self.Q_AC7(torch.stack((x[:,0],x[:,4],x[:,17],a[:,13]),1))
        Q_AC8 = self.Q_AC8(torch.stack((x[:,0],x[:,4],x[:,18],a[:,14]),1))
        Q_AC9 = self.Q_AC9(torch.stack((x[:,0],x[:,4],x[:,19],a[:,15]),1))
        out = np.cat([Q_g_b,Q_EV0,Q_EV1,Q_EV2,Q_EV3,Q_EV4,Q_AC0,Q_AC1,Q_AC2,Q_AC3,Q_AC4,Q_AC5,Q_AC6,Q_AC7,Q_AC8,Q_AC9],dim=1)
        return out