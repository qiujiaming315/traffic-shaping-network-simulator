import numpy as np
import torch
from torch import nn
from tianshou.utils.net.discrete import NoisyLinear


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class DQN(nn.Module):
    def __init__(self, state_shape, action_shape, device="cpu",
                 features_only=False, output_dim_added_layer=None, layer_init=lambda x: x):
        super().__init__()
        self.device = device
        base_cnn_output_dim = 128
        self.net = nn.Sequential(
            nn.Linear(int(np.prod(state_shape)), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, base_cnn_output_dim), nn.ReLU(inplace=True),
        )
        if not features_only:
            action_dim = int(np.prod(action_shape))
            self.net = nn.Sequential(
                self.net,
                layer_init(nn.Linear(base_cnn_output_dim, action_dim)),
            )
            self.output_dim = action_dim
        elif output_dim_added_layer is not None:
            self.net = nn.Sequential(
                self.net,
                layer_init(nn.Linear(base_cnn_output_dim, output_dim_added_layer)),
                nn.ReLU(inplace=True),
            )
            self.output_dim = output_dim_added_layer
        else:
            self.output_dim = base_cnn_output_dim

    def forward(self, obs, state=None, info={}):
        shaper_backlog_ratio_history = obs.shaper_backlog_ratio_history
        batch_size = shaper_backlog_ratio_history.shape[0]
        shaper_backlog_ratio_history = shaper_backlog_ratio_history.reshape(batch_size, -1)
        token_ratio = obs.token_ratio.reshape(batch_size, -1)
        obs = np.concatenate((shaper_backlog_ratio_history, token_ratio), axis=-1)
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float)
        logits = self.net(obs)
        return logits, state


class Rainbow(DQN):
    def __init__(self, state_shape, action_shape, num_atoms=51, noisy_std=0.5, device="cpu",
                 is_dueling=True, is_noisy=True):
        super().__init__(state_shape, action_shape, device, features_only=True)
        self.action_num = int(np.prod(action_shape))
        self.num_atoms = num_atoms

        def linear(x, y):
            if is_noisy:
                return NoisyLinear(x, y, noisy_std)
            return nn.Linear(x, y)

        self.Q = nn.Sequential(
            linear(self.output_dim, self.action_num * self.num_atoms),
        )
        self._is_dueling = is_dueling
        if self._is_dueling:
            self.V = nn.Sequential(
                linear(self.output_dim, self.num_atoms),
            )
        self.output_dim = self.action_num * self.num_atoms

    def forward(self, obs, state=None, info={}):
        r"""Mapping: x -> Z(x, \*)."""
        obs, state = super().forward(obs)
        q = self.Q(obs)
        q = q.view(-1, self.action_num, self.num_atoms)
        if self._is_dueling:
            v = self.V(obs)
            v = v.view(-1, 1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        else:
            logits = q
        probs = logits.softmax(dim=2)
        return probs, state
