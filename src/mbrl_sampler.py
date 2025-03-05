import torch
from torch.utils.data import Dataset
import numpy as np


def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]


class MBRLSampler:
    def __init__(self, rollouts, n_ensemble, batch_size, device):
        
        self.obs = np.concatenate([rollout['obs'] for rollout in rollouts], axis=0)
        self.act = np.concatenate([rollout['act'] for rollout in rollouts], axis=0) 
        self.next_obs = np.concatenate([rollout['next_obs'] for rollout in rollouts], axis=0)
        
        self.size = self.obs.shape[0]
        self.idxs = np.random.randint(self.size, size=[self.size, n_ensemble])
        self.device = device
        self.batch_size = batch_size


    def __len__(self):
        return int(np.ceil(self.size / self.batch_size))
    
    def __iter__(self):
        self.idxs = shuffle_rows(self.idxs)
        for batch_num in range(len(self)):
            batch_idxs = self.idxs[batch_num * self.batch_size : (batch_num + 1) * self.batch_size, :]

            obs = torch.tensor(self.obs[batch_idxs], device=self.device, dtype=torch.float32)
            act = torch.tensor(self.act[batch_idxs], device=self.device, dtype=torch.float32)
            next_obs = torch.tensor(self.next_obs[batch_idxs], device=self.device, dtype=torch.float32)

            yield obs, act, next_obs

    def get_val_data(self):
        self.idxs = shuffle_rows(self.idxs)
        val_obs = torch.tensor(self.obs[self.idxs[:5000]], device=self.device, dtype=torch.float32)
        val_act = torch.tensor(self.act[self.idxs[:5000]], device=self.device, dtype=torch.float32)
        val_next_obs = torch.tensor(self.next_obs[self.idxs[:5000]], device=self.device, dtype=torch.float32)
        return val_obs, val_act, val_next_obs
    
    

