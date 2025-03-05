import torch
import numpy as np
from torch import nn
import torch.functional as F
from typing import Tuple, Optional
from torch.distributions.normal import Normal

from src.utils import get_device

DEVICE = get_device()
HIDDEN_DIMENSION: int = 256
N_HIDDEN: int = 3


def tensor(x: np.array, type=torch.float32, device=DEVICE) -> torch.Tensor:
    return torch.as_tensor(x, dtype=type, device=device)


def network(
        in_dimension: int, 
        out_dimension: int, 
        hidden_dimension: int = 256, 
        n_hidden: int = 3) -> nn.Module:
    """
    Args:
        in_dimension (int): Dimension of the input layer.
        hidden_dimension (int): Dimension of the hidden layers.
        out_dimension (int): Dimension of the output layer.

    Returns:
        nn.Module: The constructed neural network model.
    """
    shapes = [in_dimension] + [hidden_dimension] * n_hidden + [out_dimension]
    layers = []
    for i in range(len(shapes) - 2):
        layers.append(nn.Linear(shapes[i], shapes[i+1]))
        layers.append(nn.Mish())
    layers.append(nn.Linear(shapes[-2], shapes[-1]))
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    def __init__(
            self,
            state_dimension: int,
            action_dimension: int,
            hidden_dimension: int = HIDDEN_DIMENSION,
            n_hidden: int = N_HIDDEN,
    ):
        super(GaussianPolicy, self).__init__()
        self.network = network(
            state_dimension, 2 * action_dimension, hidden_dimension, n_hidden
        )
        self.action_dimension = action_dimension

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Policy network. Return mean and log_std of the policy distribution

        Args:
            state (np.ndarray): The input state.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The tuple (mean, log_std) of the distribution corresponding to each action
        """
        out = self.network(state)
        mean, log_std = torch.split(out, self.action_dimension, dim=-1)
        log_std = torch.clamp(log_std, -10, 2)
        return mean, log_std
        

    def pi(self, state: torch.Tensor) -> Normal:
        """
        Computes the action distribution π(a|s) for a given state.

        Args:
            state (np.ndarray): The input state.

        Returns:
            Categorical: The action distribution.
        """
        mean, log_std = self(state)
        std = log_std.exp()
        return Normal(mean, std)

    def action(self, state: np.ndarray, eval=False) -> np.ndarray:
        """
        Selects an action based on the policy without returning the log probability.

        Args:
            state (np.ndarray): The input state.

        Returns:
            torch.Tensor: The selected action.
        """
        state = tensor(state)
        policy = self.pi(state)
        if eval:
            action = policy.mean.cpu().numpy()
        else:
            action = policy.sample().cpu().numpy()
        return action
