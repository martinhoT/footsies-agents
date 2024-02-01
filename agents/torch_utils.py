import torch
from torch import nn
from itertools import pairwise


def create_layered_network(
    input_dim: int,
    output_dim: int,
    hidden_layer_sizes: list[int],
    hidden_layer_activation: nn.Module,
):
    if hidden_layer_sizes is None:
            hidden_layer_sizes = [64, 64]

    if len(hidden_layer_sizes) == 0:
        layers = nn.Sequential(nn.Linear(input_dim, output_dim))
    else:
        layers = [
            nn.Linear(input_dim, hidden_layer_sizes[0]),
            hidden_layer_activation(),
        ]

        for hidden_layer_size_in, hidden_layer_size_out in pairwise(hidden_layer_sizes):
            layers.append(
                nn.Linear(hidden_layer_size_in, hidden_layer_size_out)
            )
            layers.append(
                hidden_layer_activation(),
            )
        
        layers.append(
            nn.Linear(hidden_layer_sizes[-1], output_dim)
        )

        layers = nn.Sequential(*layers)

    return layers


class InputClip(nn.Module):
    # Range of [5, 5] allows representing the sigmoid values of 0.01 and 0.99
    # The idea is that the network should not be too sure or unsure of the outcomes, and allow better adaptability by avoiding very small gradients at the sigmoid's tails
    # NOTE: watch out, gradients can become 0 without leaking, a leaky version has better adaptability (as in the Loss of Plasticity in Deep Continual Learning paper)
    def __init__(self, minimum: float = -5, maximum: float = 5, leaky_coef: float = 0):
        """Clip input into range"""
        super().__init__()
        self.minimum = minimum
        self.maximum = maximum
        self.leaky_coef = leaky_coef

    def forward(self, x: torch.Tensor):
        return torch.clip(
            x,
            min=self.minimum + self.leaky_coef * (x - 5),
            max=self.maximum + self.leaky_coef * (x + 5),
        )


class ProbabilityDistribution(nn.Module):
    def __init__(self):
        """Makes input sum to 1"""
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x / torch.sum(x)


class DebugStoreRecent(nn.Module):
    def __init__(self):
        """Store the most recent input"""
        super().__init__()
        self.stored = None

    def forward(self, x: torch.Tensor):
        self.stored = x
        return x
