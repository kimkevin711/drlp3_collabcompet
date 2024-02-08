import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=136, fc2_units=66, fc3_units=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc1_bn = nn.BatchNorm1d(fc1_units)
        self.fc1_dp = nn.Dropout(p=0.1)
        
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc2_bn = nn.BatchNorm1d(fc2_units)
        self.fc2_dp = nn.Dropout(p=0.1)
        
#         self.fc3 = nn.Linear(fc2_units, fc3_units)
#         self.fc3_bn = nn.BatchNorm1d(fc3_units)
#         self.fc3_dp = nn.Dropout(p=0.3)
        
        self.fc_last = nn.Linear(fc2_units, action_size)
#         self.fc_last = nn.Linear(fc3_units, action_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc_last.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""

        x = self.fc1_dp(F.relu(self.fc1(state)))
        x = self.fc2_dp(F.relu(self.fc2(x)))
#         x = self.fc3_dp(F.relu(self.fc3(x)))
        
        return torch.tanh(self.fc_last(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

#     def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
    def __init__(self, state_size, action_size, seed, fcs1_units=136, fc2_units = 66, fc3_units=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc_dp = nn.Dropout(p=0.1)
        
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fcs1_bn = nn.BatchNorm1d(fcs1_units)
        
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)  
        self.fc2_bn = nn.BatchNorm1d(fc2_units)
        
#         self.fc3 = nn.Linear(fc2_units, fc3_units)  
#         self.fc3_bn = nn.BatchNorm1d(fc3_units)

        self.fc_last = nn.Linear(fc2_units, 1)
#         self.fc_last = nn.Linear(fc3_units, 1)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
#         self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc_last.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        
        xs = F.relu(self.fcs1_bn(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        
#         xs = self.fc_dp(F.relu(self.fcs1_bn(self.fcs1(state))))   # batch normalization
#         x = torch.cat((xs, action), dim=1)
#         x = self.fc_dp(F.relu(self.fc2_bn(self.fc2(x))))          # batch normalization
#         x = self.fc_dp(F.relu(self.fc3_bn(self.fc3(x))))          # batch normalization
        return self.fc_last(x)
