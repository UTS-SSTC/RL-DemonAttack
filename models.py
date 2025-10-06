import torch.nn as nn


class DQN(nn.Module):
    """
    Nature DQN convnet for 4x84x84 input.
    """

    def __init__(self, n_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


class DuelingDQN(nn.Module):
    """
    Dueling head variant.
    """

    def __init__(self, n_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        )
        self.adv = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )
        self.val = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        feat = self.features(x)
        adv = self.adv(feat)
        val = self.val(feat)
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q
