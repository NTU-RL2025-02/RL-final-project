import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits
        return self.act_limit * self.pi(obs)


class MLPClassifier(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, device):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Sigmoid)
        self.device = device

    def forward(self, obs):
        # Return output from network scaled to action space limits
        return self.pi(obs).to(self.device)


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [1], activation, nn.Sigmoid
        )

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class MLP(nn.Module):

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(
            device
        )
        self.pi_safe = MLPClassifier(obs_dim, 1, (128, 128), activation, device).to(
            device
        )
        self.device = device

    def act(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.pi(obs).cpu().numpy()

    def classify(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.pi_safe(obs).cpu().numpy().squeeze()


class Ensemble(nn.Module):
    # Multiple policies
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
        num_nets=5,
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        self.num_nets = num_nets
        self.device = device
        # build policy and value functions
        self.pis = nn.ModuleList([
            MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit).to(device)
            for _ in range(num_nets)
        ])
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation).to(device)

    def act(self, obs, i=-1):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if i >= 0:  # optionally, only use one of the nets.
                return self.pis[i](obs).cpu().numpy()
            a = torch.mean(torch.stack([pi(obs) for pi in self.pis]), dim=0)
            return a.cpu().numpy()

    def safety(self, obs, act):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = torch.min(self.q1(obs, act), self.q2(obs, act))
            return q.cpu().numpy().squeeze()

    def variance(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            # 返回 ensemble 中各個 policy 的輸出方差
            return torch.std(torch.stack([pi(obs) for pi in self.pis]), dim=0).mean()
