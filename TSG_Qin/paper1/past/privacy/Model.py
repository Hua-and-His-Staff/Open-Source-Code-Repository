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
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2.))
        module.bias.data.zero_()
        
    elif classname == 'Conv2d':
        nn.init.orthogonal_(module.weight, scale=np.sqrt(2.))
        module.bias.data.zero_()
    

    
    
    
class Actor(nn.Module):
    def __init__(self, num_observations,num_continuous_actions,num_discrete_actions):
        """ Basic MLP Actor-Critic Network for Linear Time Invariant System
        With Infinite Time Horizon
        Args:
            num_actions (int): the number of available discrete actions
        """
        super().__init__()
        self.hiden = nn.Sequential(nn.Linear(num_observations,64),
                                  nn.Tanh(),
                                  nn.Linear(64,128),
                                  nn.Tanh(),
                                  nn.Linear(128,128),                                
                                  nn.Tanh(),
                                  nn.Linear(128,64),
                                  nn.Tanh())
        
        self.mu = nn.Sequential(nn.Linear(64,num_continuous_actions),
                                  nn.Sigmoid())   
        
        self.sigma = nn.Sequential(nn.Linear(64,num_continuous_actions),
                                  nn.Sigmoid())
        
        self.AC_prob = nn.Sequential(nn.Linear(num_observations,64),
                                  nn.Tanh(),
                                  nn.Linear(64,128),
                                  nn.Tanh(),
                                  nn.Linear(128,128),                                
                                  nn.Tanh(),
                                  nn.Linear(128,128),
                                  nn.Tanh(),
                                  nn.Linear(128,num_discrete_actions),
                                  nn.Sigmoid())        

        self.num_observations=num_observations
        
        
        # parameter initialization
        self.apply(Grid_initializer)
      
    

    def forward(self, x_in):
        """ Module forward pass
        Args:
            x_in (Variable):  input, shaped [N x self.num_observations]
        Returns:
            pi (Variable): action , shaped [N x self.num_actions]
        """
        hiden = self.hiden(x_in)
        mu_out = self.mu(hiden)
        sigma_out = self.sigma(hiden)
        prob_out = self.AC_prob(x_in)

        return mu_out, sigma_out, prob_out
    
    
class Critic(nn.Module):
    def __init__(self, num_observations,num_AC):
        """ Basic MLP Actor-Critic Network for Linear Time Invariant System
        With Infinite Time Horizon
        Args:
            num_actions (int): the number of available discrete actions
        """
        super().__init__()
        
        self.gen_BES_EV_v = nn.Sequential(nn.Linear(num_observations,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),
                                  nn.Tanh(),
                                  nn.Linear(64,64),                                
                                  nn.Tanh(),
                                  nn.Linear(64, 3))
        self.AC_v = nn.Sequential(nn.Linear(num_observations,64),
                                  nn.Tanh(),
                                  nn.Linear(64,128),
                                  nn.Tanh(),
                                  nn.Linear(128,128),                                
                                  nn.Tanh(),
                                  nn.Linear(128, num_AC)
                                 )
        # parameter initialization
        self.apply(Grid_initializer)
      
    

    def forward(self, x_in):
        """ Module forward pass
        Args:
            x_in (Variable):  input, shaped [N x self.num_observations]
        Returns:
            v (Variable): value predictions, shaped [N x 3]
        """
        gen_BES_EV_values = self.gen_BES_EV_v(x_in)
        AC_values = self.AC_v(x_in)
        return torch.cat((gen_BES_EV_values,AC_values),dim=1)