import numpy as np

from src.cem import CEMOptimizer

import torch
import torch.nn.functional as F
import einops


def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]


class MPC:
    optimizers = {"CEM": CEMOptimizer}

    def __init__(self, 
                 observation_space,
                 action_space,
                 obs_cost_fn,
                 act_cost_fn,
                 dynamics_model,

                 plan_hor,
                 n_particles,
                 device,

                 max_iters=5,
                 popsize=500,
                 num_elites=50,
                 alpha=0.1,
                 ):
        """Creates class instance.
        """
        self.dO, self.dU = observation_space.shape[0], action_space.shape[0]
        self.ac_ub, self.ac_lb = action_space.high, action_space.low

        self.plan_hor = plan_hor
        self.n_particles = n_particles
        self.device = device

        self.obs_cost_fn = obs_cost_fn
        self.act_cost_fn = act_cost_fn
        self.dynamics_model = dynamics_model

        # Create action sequence optimizer
        self.optimizer = CEMOptimizer(
            sol_dim=self.plan_hor * self.dU,
            lower_bound=torch.tensor(np.tile(self.ac_lb, [self.plan_hor]), device=device),
            upper_bound=torch.tensor(np.tile(self.ac_ub, [self.plan_hor]), device=device),
            cost_function=self._compile_cost,
            max_iters=max_iters,
            popsize=popsize,
            num_elites=num_elites,
            alpha=alpha,
        )

        # Controller state variables
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.plan_hor])

    def reset(self):
        """Resets this controller (clears previous solution, calls all update functions).

        Returns: None
        """
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub) / 2, [self.plan_hor])
        self.optimizer.reset()

    def action(self, obs):
        return self.act(obs, None)
    
    def eval(self):
        self.dynamics_model.eval()

    def train(self):
        self.dynamics_model.train()

    def act(self, obs, t, get_pred_cost=False):
        """Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation
            t: The current timestep
            get_pred_cost: If True, returns the predicted cost for the action sequence found by
                the internal optimizer.

        Returns: An action (and possibly the predicted cost)
        """
        self.sy_cur_obs = torch.tensor(obs, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            soln = self.optimizer.obtain_solution(
                torch.tensor(self.prev_sol, device=self.device, dtype=torch.float32), 
                torch.tensor(self.init_var, device=self.device, dtype=torch.float32)).cpu().numpy()
        
        self.prev_sol = np.concatenate([np.copy(soln)[self.dU:], np.zeros(self.dU)])
    
        action = einops.rearrange(soln, '(h a) -> h a', h=self.plan_hor)

        return action[0]

    
    @torch.no_grad()
    def _compile_cost(self, ac_seqs):

        actions = einops.rearrange(ac_seqs, 'n (h a) -> n h a', h=self.plan_hor)
        costs = self.dynamics_model.compute_cost(
            self.sy_cur_obs, 
            actions, 
            self.obs_cost_fn,
            self.act_cost_fn,
            self.n_particles)
        return costs
