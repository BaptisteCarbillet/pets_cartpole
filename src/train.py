import torch
import torch.nn as nn
import numpy as np

import src.cartpole_env
from src.mpc import MPC
from src.mbrl_utils import sample_rollout
from src.cartpole_env import CartpoleConfigModule
from src.mbrl_sampler import MBRLSampler

from torch.optim import Adam
from src.model import EnsembleDynamicsModel
from src.utils import get_device, set_seed
from tqdm import trange
from loguru import logger
import gym

################################## Environment #########################################

DEVICE = get_device()
ENVIRONMENT_NAME: str='MBRLCartpole-v0'

# torch related defaults
DEVICE = get_device()
torch.set_default_dtype(torch.float32)

# Use random seeds for reproducibility
SEED = 42
set_seed(SEED)
env = gym.make(ENVIRONMENT_NAME)

# get the state and action dimensions
action_dimension = env.action_space.shape[0]
state_dimension = env.observation_space.shape[0]

logger.info(f'Action Dimension: {action_dimension}')
logger.info(f'Action High: {env.action_space.high}')
logger.info(f'Action Low: {env.action_space.low}')
logger.info(f'State Dimension: {state_dimension}')

################################## Hyper-parameters #########################################

EPOCHS = 150
EVAL_FREQ = 30
TASK_HORIZON = 200

plan_hor = 25
n_particles = 10
batch_size = 32
n_ensemble = 5
maxiters = 5
popsize = 100
num_elites = 10

################################### Cost Functions ###########################################

sampler = MBRLSampler(torch.load('data.pkl'), n_ensemble, batch_size, DEVICE)


rollouts = torch.load('data.pkl')
all_obs = np.concatenate([rollout['obs'] for rollout in rollouts], axis=0)
all_act = np.concatenate([rollout['act'] for rollout in rollouts], axis=0)
all_next_obs = np.concatenate([rollout['next_obs'] for rollout in rollouts], axis=0)

config = CartpoleConfigModule(DEVICE)
dynamics_model = EnsembleDynamicsModel(state_dimension, action_dimension, n_ensemble).to(DEVICE)
optimizer = Adam(dynamics_model.params.values(), 1e-3, weight_decay=1e-4)
policy = MPC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    obs_cost_fn=config.obs_cost_fn,
    act_cost_fn=config.ac_cost_fn,
    dynamics_model=dynamics_model,
    plan_hor=plan_hor,
    n_particles=n_particles,
    max_iters=maxiters,
    popsize=popsize,
    num_elites=num_elites,
    alpha=0.1,
    device=DEVICE
)

data_len = all_obs.shape[0]

epoch_range = trange(EPOCHS, unit="epoch(s)", desc="Network training")
num_batch = int(np.ceil(data_len / batch_size))
result = None
rews = []

for epoch in epoch_range:

    for obs, act, next_obs in sampler:

        #compute NLL loss and update the dynamics model
        mean, log_std = dynamics_model(obs, act)
        
        std = torch.exp(log_std)
        distribution = torch.distributions.Normal(mean, std)
        nll_loss = -distribution.log_prob(next_obs).mean()
        optimizer.zero_grad()
        nll_loss.backward()
        optimizer.step()
        
    # Compute validation MSE loss
    val_obs, val_act, val_next_obs = sampler.get_val_data()
    
    mean, _ = dynamics_model(val_obs, val_act)
    mse_losses = ((mean - val_next_obs) ** 2).mean()

    epoch_range.set_postfix({
        "Training loss": mse_losses.item(),
        'Reward': result
    })

    # Sample an eval rollout. If you are not using a GPU you should comment this out and only run eval once
    if (epoch + 1) % EVAL_FREQ == 0:
        info = sample_rollout(
            env,
            TASK_HORIZON,
            policy=policy,
        )
        result = info['reward_sum']
        rews.append(result)

torch.save(dynamics_model.state_dict(), 'pets_checkpoint.pth')
