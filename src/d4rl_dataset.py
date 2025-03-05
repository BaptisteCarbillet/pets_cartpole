import torch
import numpy as np
from torch.utils.data import Dataset

class D4RLDataset(Dataset):
    def __init__(self, data, device):
        
        self.state = torch.tensor(data['observations'], dtype=torch.float32, device=device)
        self.action = torch.tensor(data['actions'], dtype=torch.float32, device=device)
        self.next_state = torch.tensor(data['next_observations'], dtype=torch.float32, device=device)
        self.reward = torch.tensor(data['rewards'], dtype=torch.float32, device=device)
        self.not_done = 1. - torch.tensor(data['terminals'], dtype=torch.float32, device=device)

        self.size = self.state.shape[0]


    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'state': self.state[idx],
            'action': self.action[idx],
            'next_state': self.next_state[idx],
            'reward': self.reward[idx],
            'not_done': self.not_done[idx],
        }
    

class D4RLSampler:
    def __init__(self, data, batch_size, device):
        
        self.state = torch.tensor(data['observations'], dtype=torch.float32, device=device)
        self.action = torch.tensor(data['actions'], dtype=torch.float32, device=device)
        self.next_state = torch.tensor(data['next_observations'], dtype=torch.float32, device=device)
        self.reward = torch.tensor(data['rewards'], dtype=torch.float32, device=device)
        self.not_done = 1. - torch.tensor(data['terminals'], dtype=torch.float32, device=device)

        self.size = self.state.shape[0]
        self.batch_size = batch_size

    def __iter__(self):
        for batch_num in range(len(self)):
            batch_idxs = np.random.randint(0, self.size, size=self.batch_size)

            state = self.state[batch_idxs]
            action = self.action[batch_idxs]
            next_state = self.next_state[batch_idxs]
            reward = self.reward[batch_idxs]
            not_done = self.not_done[batch_idxs]

            batch = {
                'state': state,
                'action': action,
                'next_state': next_state,
                'reward': reward,
                'not_done': not_done,
            }

            yield batch

    def __len__(self):
        return int(np.ceil(self.size / self.batch_size))
    