import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
    

def Grid_initializer(module):
    """ Parameter initializer for Atari models
    Initializes Linear, Conv2d, and LSTM weights.
    """
    classname = module.__class__.__name__

    if classname == 'Linear':
        
        #nn.init.orthogonal_(module.weight, gain=np.sqrt(2.))
        nn.init.normal_(module.weight, mean=0.0, std=1.0)
        module.bias.data.zero_()
        
    elif classname == 'Conv2d':
        nn.init.orthogonal_(module.weight, scale=np.sqrt(2.))
        module.bias.data.zero_()
    

    
    
    
class Network(nn.Module):
    def __init__(self, GRU_input_size, GRU_hidden_size, MLP_input_size, num_actions, num_values):
        """ Basic MLP Actor-Critic Network for Linear Time Invariant System
        With Infinite Time Horizon
        Args:
            num_actions (int): the number of available discrete actions
        """
        super().__init__()
        
        self.GRU = nn.GRUCell(GRU_input_size, GRU_hidden_size)
        
        
        
        self.mu = nn.Sequential(nn.Linear(GRU_hidden_size+MLP_input_size,128),
                                  nn.Tanh(),
                                  nn.Linear(128,128),
                                  nn.Tanh(),
                                  nn.Linear(128,128),                                
                                  nn.Tanh(),
                                  nn.Linear(128,num_actions),
                                  nn.Sigmoid())   
        
        self.sigma = nn.Sequential(nn.Linear(GRU_hidden_size+MLP_input_size,128),
                                  nn.Tanh(),
                                  nn.Linear(128,128),
                                  nn.Tanh(),
                                  nn.Linear(128,128),
                                  nn.Tanh(),
                                  nn.Linear(128,num_actions),
                                  nn.Sigmoid())
        
        self.value = nn.Sequential(nn.Linear(GRU_hidden_size+MLP_input_size,128),
                                  nn.Tanh(),
                                  nn.Linear(128,128),
                                  nn.Tanh(),
                                  nn.Linear(128,128),                                
                                  nn.Tanh(),
                                  nn.Linear(128, 1))
        
        # parameter initialization
        self.apply(Grid_initializer)
      
    

    def forward(self, obs_GRU, obs_MLP, h):
        """ Module forward pass
        Args:
            x_in (Variable):  input, shaped [N x self.num_observations]
        Returns:
            pi (Variable): action , shaped [N x self.num_actions]
        """
        
        hidden = self.GRU(obs_GRU, h)
        features = torch.cat([hidden,obs_MLP],dim=1) 
        mus = self.mu(features)
        sigmas = self.sigma(features)
        values = self.value(features)

        return mus, sigmas, values, hidden
    
    