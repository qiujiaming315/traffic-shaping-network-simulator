import numpy as np
import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, state_shape, action_shape, device="cpu"):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(state_shape)), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, int(np.prod(action_shape))),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, device=self.device, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state
