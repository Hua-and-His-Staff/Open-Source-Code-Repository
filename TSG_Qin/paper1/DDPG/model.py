
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
        nn.init.normal_(module.weight, mean=0.0, std=1.0)
        module.bias.data.zero_()
        
    elif classname == 'Conv2d':
        nn.init.normal_(module.weight, mean=0.0, std=1.0)
        module.bias.data.zero_()
    

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, num_observations, num_actions, init_w=3e-3):
        super(Actor, self).__init__()
        '''
        self.DG = nn.Sequential(nn.Linear(num_observations,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),                                
                                  nn.Tanh(),
                                  nn.Linear(64,1),
                                  nn.Sigmoid())
        self.EV = nn.Sequential(nn.Linear(num_observations,64),
                                  nn.Tanh(),
                                  nn.Linear(64,128),
                                  nn.Tanh(),
                                  nn.Linear(128,128),                                
                                  nn.Tanh(),
                                  nn.Linear(128,128),
                                  nn.Tanh(),
                                  nn.Linear(128,5),
                                  nn.Sigmoid())  
        self.AC = nn.Sequential(nn.Linear(num_observations,64),
                                  nn.Tanh(),
                                  nn.Linear(64,128),
                                  nn.Tanh(),
                                  nn.Linear(128,128),                                
                                  nn.Tanh(),
                                  nn.Linear(128,128),
                                  nn.Tanh(),
                                  nn.Linear(128,10),
                                  nn.Sigmoid())
        '''
        self.u = nn.Sequential(nn.Linear(num_observations,64),
                                  nn.Tanh(),
                                  nn.Linear(64,128),
                                  nn.Tanh(),
                                  nn.Linear(128,128),                                
                                  nn.Tanh(),
                                  nn.Linear(128,128),
                                  nn.Tanh(),
                                  nn.Linear(128,num_actions),
                                  nn.Sigmoid())
        self.apply(Grid_initializer)
    
    #def init_weights(self, init_w):
    #    self.input.weight.data = fanin_init(self.input.weight.data.size())
        #self.hidden1.weight.data = fanin_init(self.hidden1.weight.data.size())
        #self.hidden2.weight.data = fanin_init(self.hidden2.weight.data.size())
    #    self.output.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        #uDG = self.DG(x)
        #uEV = self.EV(x)
        #uAC = self.AC(x)
        #torch.cat([uDG,uEV,uAC],1)
        return self.u(x)
        
class Critic(nn.Module):
    def __init__(self, num_observations, num_actions, num_values, init_w=3e-3):
        super(Critic, self).__init__()
        '''
        self.V = nn.Sequential(nn.Linear(num_observations,64),
                                  nn.Tanh(),
                                  nn.Linear(64,128),
                                  nn.Tanh(),
                                  nn.Linear(128, 64),
                                  nn.Tanh())
        
        self.bias = nn.Sequential(nn.Linear(num_actions,64),
                                  nn.Tanh(),
                                  nn.Linear(64,128),
                                  nn.Tanh(),
                                  nn.Linear(128, 64),
                                  nn.Tanh())
        '''
        
        self.Q = nn.Sequential(nn.Linear(num_observations + num_actions,64),                                
                                  nn.Tanh(),
                                  nn.Linear(64,128),                                
                                  nn.Tanh(),
                                  nn.Linear(128,128),                                
                                  nn.Tanh(),
                                  nn.Linear(128,128),                                
                                  nn.Tanh(),  
                                  nn.Linear(128,128),                                
                                  nn.Tanh(),
                                  nn.Linear(128, num_values))
        
        self.apply(Grid_initializer)
    
    #def init_weights(self, init_w):
    #    self.input1.weight.data = fanin_init(self.input1.weight.data.size())
    #    self.input2.weight.data = fanin_init(self.input2.weight.data.size())
    #    self.hidden1.weight.data = fanin_init(self.hidden1.weight.data.size())     
    #    self.output.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        xa = torch.cat([x,a],1)
        #value = self.V(x)
        #bias = self.bias(a)
        out = self.Q(xa)
        return out