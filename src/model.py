from src.networks import network
import torch
import torch.nn as nn
from typing import Tuple

from src.utils import get_device

DEVICE = get_device()

class DynamicsModel(nn.Module):
    def __init__(self, 
                 state_dimension: int, 
                 action_dimension: int,
                 min_log_std: float = -5,
                 max_log_std: float = 1,
                 ):
        super(DynamicsModel, self).__init__()
        self.network = network(state_dimension + action_dimension, 2 * state_dimension, hidden_dimension=128,n_hidden=2)
        
        self.state_dimension = state_dimension
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the dynamics network. Should return mean and log_std of the next state distribution

        Args:
            state (torch.Tensor): The input state.
            action (torch.Tensor): The input action.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The tuple (mean, log_std) of the distribution
        """

        # predict the mean and log_std of the next state distribution
        out = self.network(torch.cat([state, action], dim=-1))
        mean, log_std = torch.split(out,self.state_dimension,dim=-1)
        log_std = log_std.clamp(min=self.min_log_std, max=self.max_log_std)
        
        return mean, log_std
        



from torch import vmap 
from torch.func import functional_call
import copy
from torch.func import stack_module_state


class EnsembleDynamicsModel(nn.Module):
    def __init__(self, state_dimension: int, action_dimension: int, n_ensemble: int):
        super(EnsembleDynamicsModel, self).__init__()
        self.num_nets = n_ensemble
        self.models = ([DynamicsModel(state_dimension, action_dimension,min_log_std=-5,max_log_std=1).to(DEVICE) for _ in range(n_ensemble)])
        
        self.params, self.buffers = stack_module_state(self.models)
        
        self.base_model = copy.deepcopy(self.models[0])
        
        
    
    def fmodel(self,params,buffers, x,y):
        return functional_call(self.base_model, (params, buffers), (x,y))
        
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the dynamics network. Return mean and log_std of the next state distribution for each model in the ensemble

        Args:
            state (torch.Tensor): The input state, shape (B, n_ensemble, S)
            action (torch.Tensor): The input action, shape (B, n_ensemble, A)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The tuple (mean, log_std) of the distributions where each have shape (B, n_ensemble, S)
        """

        
        if state.device != DEVICE:
            state = state.to(DEVICE)
            
        mean, log_std = vmap(self.fmodel)(self.params, self.buffers, state.permute(1,0,2), action.permute(1,0,2))
        

        return mean.permute(1,0,2), log_std.permute(1,0,2)
    def compute_cost(
            self, 
            state: torch.Tensor, 
            actions: torch.Tensor,
            obs_cost_fn,
            act_cost_fn,
            n_particles: int = 20,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a state and a 

        Args:
            state (torch.Tensor): The input state, shape (S,)
            actions (torch.Tensor): The action sequence candidates, shape (N, H, A)
            obs_cost_fn: A function which takes in a batch of states and returns the cost of each one
            act_cost_fn: A function which takes in a batch of actions and returns the cost of each one
            n_particles (int): how many particles to sample for each action sequence

        Returns:
            torch.Tensor: Expected cost for each action candidate, shape (N,)
        """
        n_candidates, horizon, _ = actions.shape
        state_dim = 4 
        
        trajectory = torch.zeros(n_particles,n_candidates,horizon,state_dim)
        trajectory[:,:,0,:] = state
        costs = torch.zeros(n_candidates).to(DEVICE)
        
        for i in range(horizon-1):
            
            s_t = trajectory[:,:,i,:].reshape(-1,state_dim).unsqueeze(1).repeat(1,self.num_nets,1) #shape (n_particles, n_candidates, state_dim) at first to shape : (n_particles*n_candidates, n_ensemble,state_dim)
            a_t = actions[:,i,:].unsqueeze(1).repeat(n_particles,self.num_nets,1) #shape (n_candidates, action_dim) to shape : (n_particles*n_candidates, n_ensemble, action_dim)
            
            mean, log_std = self.forward(s_t, a_t)
            mean,log_std = mean.mean(dim=1),log_std.mean(dim=1).clamp(-5,1) #shape (n_particles*n_candidates, state_dim)
            #Idk why the clamping doesnt consistantly work with vmap, especially when evaluating the policy
            
            std = torch.exp(log_std)
            
            
                
            distribution = torch.distributions.Normal(mean,std)
            next_state = distribution.sample().reshape(n_particles,n_candidates,state_dim)
            trajectory[:,:,i+1,:] = next_state
            
            
            obs_cost = obs_cost_fn(next_state).mean(dim=0)
            action_cost = act_cost_fn(actions[:,i,:])
            costs += obs_cost + action_cost
            
            if torch.isnan(costs).sum() > 0:
                
                costs[torch.isnan(costs)] = 1e6
                
        
            
        
        return costs
