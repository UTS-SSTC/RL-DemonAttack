"""
Deep Q-Network models for Atari DemonAttack.

Implements Nature DQN and Dueling DQN architectures optimized for
84x84 grayscale frame-stacked observations.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Nature DQN convolutional network.

    Architecture designed for 4-frame stacked 84x84 grayscale observations.
    Uses three convolutional layers followed by two fully connected layers.

    Args:
        n_actions: Number of discrete actions in the environment.
    """

    def __init__(self, n_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, 4, 84, 84].

        Returns:
            Q-values for each action with shape [batch_size, n_actions].
        """
        x = self.features(x)
        return self.head(x)


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture.

    Separates value and advantage streams for improved learning stability.
    Uses the same convolutional backbone as Nature DQN but splits into
    separate value and advantage heads.

    Args:
        n_actions: Number of discrete actions in the environment.
    """

    def __init__(self, n_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.adv = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )
        self.val = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        """
        Forward pass through the dueling architecture.

        Computes Q(s,a) = V(s) + A(s,a) - mean(A(s,a)).

        Args:
            x: Input tensor of shape [batch_size, 4, 84, 84].

        Returns:
            Q-values for each action with shape [batch_size, n_actions].
        """
        feat = self.features(x)
        adv = self.adv(feat)
        val = self.val(feat)
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q


class NoisyLinear(nn.Module):
    """
    Noisy Linear layer for exploration.

    Implements noisy networks for exploration as described in
    "Noisy Networks for Exploration" (Fortunato et al., 2017).

    Args:
        in_features: Size of input features.
        out_features: Size of output features.
        std_init: Initial standard deviation for noise.
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """Reset noise buffers."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        weight_eps: torch.Tensor = self.weight_epsilon  # type: ignore[assignment]
        bias_eps: torch.Tensor = self.bias_epsilon  # type: ignore[assignment]
        weight_eps.copy_(torch.outer(epsilon_out, epsilon_in))
        bias_eps.copy_(epsilon_out)

    def _scale_noise(self, size: int):
        """Scale noise using factorized Gaussian noise."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        """
        Forward pass with noisy weights.

        Args:
            x: Input tensor.

        Returns:
            Output with noisy transformation applied.
        """
        if self.training:
            weight_epsilon: torch.Tensor = self.weight_epsilon  # type: ignore[assignment]
            bias_epsilon: torch.Tensor = self.bias_epsilon  # type: ignore[assignment]
            weight = self.weight_mu + self.weight_sigma * weight_epsilon
            bias = self.bias_mu + self.bias_sigma * bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class NoisyDQN(nn.Module):
    """
    DQN with Noisy Networks for exploration.

    Replaces epsilon-greedy exploration with learned parametric noise.
    Better exploration in later stages of training.

    Args:
        n_actions: Number of discrete actions in the environment.
    """

    def __init__(self, n_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            NoisyLinear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            NoisyLinear(512, n_actions),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, 4, 84, 84].

        Returns:
            Q-values for each action with shape [batch_size, n_actions].
        """
        x = self.features(x)
        return self.head(x)

    def reset_noise(self):
        """Reset noise in all noisy layers."""
        for module in self.head:
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class NoisyDuelingDQN(nn.Module):
    """
    Dueling DQN with Noisy Networks.

    Combines dueling architecture with noisy networks for both
    improved value estimation and better exploration.

    Args:
        n_actions: Number of discrete actions in the environment.
    """

    def __init__(self, n_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.adv = nn.Sequential(
            nn.Flatten(),
            NoisyLinear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            NoisyLinear(512, n_actions),
        )
        self.val = nn.Sequential(
            nn.Flatten(),
            NoisyLinear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            NoisyLinear(512, 1),
        )

    def forward(self, x):
        """
        Forward pass through the dueling architecture.

        Computes Q(s,a) = V(s) + A(s,a) - mean(A(s,a)).

        Args:
            x: Input tensor of shape [batch_size, 4, 84, 84].

        Returns:
            Q-values for each action with shape [batch_size, n_actions].
        """
        feat = self.features(x)
        adv = self.adv(feat)
        val = self.val(feat)
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q

    def reset_noise(self):
        """Reset noise in all noisy layers."""
        for module in list(self.adv) + list(self.val):
            if isinstance(module, NoisyLinear):
                module.reset_noise()
