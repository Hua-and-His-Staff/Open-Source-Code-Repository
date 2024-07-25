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
    

    
    
    
class Actor(nn.Module):
    def __init__(self, num_observations,num_actions):
        """ Basic MLP Actor-Critic Network for Linear Time Invariant System
        With Infinite Time Horizon
        Args:
            num_actions (int): the number of available discrete actions
        """
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(num_observations,64),
                                  nn.Tanh(),
                                  nn.Linear(64,128),
                                  nn.Tanh(),
                                  nn.Linear(128,128),                                
                                  nn.Tanh(),
                                  nn.Linear(128,128),
                                  nn.Tanh(),
                                  nn.Linear(128,num_actions),
                                  nn.Sigmoid())   
        
        self.sigma = nn.Sequential(nn.Linear(num_observations,64),
                                  nn.Tanh(),
                                  nn.Linear(64,128),
                                  nn.Tanh(),
                                  nn.Linear(128,128),
                                  nn.Tanh(),
                                  nn.Linear(128,num_actions),
                                  nn.Sigmoid())
        
        # parameter initialization
        self.apply(Grid_initializer)
      
    

    def forward(self, x_in):
        """ Module forward pass
        Args:
            x_in (Variable):  input, shaped [N x self.num_observations]
        Returns:
            pi (Variable): action , shaped [N x self.num_actions]
        """
        mu_out = self.mu(x_in)
        sigma_out = self.sigma(x_in)

        return mu_out, sigma_out
    
    
class Critic(nn.Module):
    def __init__(self, num_observations,num_value):
        """ Basic MLP Actor-Critic Network for Linear Time Invariant System
        With Infinite Time Horizon
        Args:
            num_actions (int): the number of available discrete actions
        """
        super().__init__()
        
        self.values = nn.Sequential(nn.Linear(num_observations,64),
                                  nn.Tanh(),
                                  nn.Linear(64,128),
                                  nn.Tanh(),
                                  nn.Linear(128,128),                                
                                  nn.Tanh(),
                                  nn.Linear(128,128),                                
                                  nn.Tanh(),
                                  nn.Linear(128, num_value))
        # parameter initialization
        self.apply(Grid_initializer)
      
    

    def forward(self, x_in):
        """ Module forward pass
        Args:
            x_in (Variable):  input, shaped [N x self.num_observations]
        Returns:
            v (Variable): value predictions, shaped [N x 3]
        """
        return self.values(x_in)