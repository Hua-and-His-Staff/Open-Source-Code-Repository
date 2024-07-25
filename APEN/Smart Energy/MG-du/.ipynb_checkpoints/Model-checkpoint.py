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
    

    
    
    
class GridMLP(nn.Module):
    def __init__(self, num_observations,num_actions):
        """ Basic MLP Actor-Critic Network for Linear Time Invariant System
        With Infinite Time Horizon
        Args:
            num_actions (int): the number of available discrete actions
        """
        super().__init__()
        
        self.input_feature = nn.Sequential(nn.Linear(num_observations,128),
                                  nn.ELU(),
                                  nn.Linear(128,128),
                                  nn.ELU(),
                                  nn.Linear(128,128),
                                  nn.ELU(),
                                  nn.Linear(128,256),
                                  nn.LayerNorm(256),
                                  nn.ELU())
        
        self.fc = nn.Sequential(nn.Linear(256,256),
                                  nn.ELU(),
                                  nn.Linear(256,128),
                                  nn.ELU(),
                                  nn.Linear(128,128),
                                  nn.ELU(),
                                  nn.Linear(128,128),
                                  nn.LayerNorm(128),
                                  nn.ELU())
        
        self.mu = nn.Sequential(nn.Linear(128,128),
                                  nn.ELU(),
                                  nn.Linear(128,128),
                                  nn.ELU(),
                                  nn.Linear(128, num_actions),
                                  nn.LayerNorm(num_actions),
                                  nn.Tanh())
        self.theta = nn.Sequential(nn.Linear(128,128),
                                  nn.ELU(),
                                  nn.Linear(128,128),
                                  nn.ELU(),
                                  nn.Linear(128, num_actions*num_observations),
                                  nn.LayerNorm(num_actions*num_observations),
                                  nn.Tanh())
        
        
        self.sigma = nn.Sequential(nn.Linear(128,128),
                                  nn.ELU(),
                                  nn.Linear(128,128),
                                  nn.ELU(),
                                  nn.Linear(128, num_actions),
                                  nn.LayerNorm(num_actions),
                                  nn.Sigmoid())
        
        self.v = nn.Sequential(nn.Linear(128,128),
                                  nn.ELU(),
                                  nn.Linear(128,128),
                                  nn.ELU(),
                                  nn.Linear(128,128),
                                  nn.ELU(),
                                  nn.Linear(128, 1))
        
        self.num_observations=num_observations
        self.num_actions = num_actions
        
        
        # parameter initialization
        self.apply(Grid_initializer)
      
    

    def forward(self, x_in):
        """ Module forward pass
        Args:
            x_in (Variable):  input, shaped [N x self.num_observations]
        Returns:
            pi (Variable): action , shaped [N x self.num_actions]
            v (Variable): value predictions, shaped [N x 1]
        """

        input_feature = self.input_feature(x_in)
        fc_out=self.fc(input_feature)
        mu_out = self.mu(fc_out)
        theta_out=torch.reshape(self.theta(fc_out),(-1,self.num_actions,self.num_observations))
        sigma_out = self.sigma(fc_out)
        v_out = self.v(fc_out)

        return mu_out+torch.matmul(theta_out,torch.unsqueeze(x_in,-1)).squeeze(-1), sigma_out, v_out