from torch import nn

class CartePoleNet(nn.Module):
    """Policy network: observation -> 128 hidden units (ReLU) -> action logits.

    Outputs raw logits (no softmax); apply softmax / log_softmax outside.
    """

    def __init__(self, actions: int, observation_space: int):
        """
        Args:
            actions: number of discrete actions (output logits).
            observation_space: size of the observation vector (input features).
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(observation_space, 128),
            nn.ReLU(),
            nn.Linear(128, actions)
        )

    def forward(self, x):
        return self.model(x)