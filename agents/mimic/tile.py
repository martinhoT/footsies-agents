import torch
import numpy as np
from dataclasses import dataclass
from enum import Enum
from functools import reduce


# Smooth tilings? Instead of 0s and 1s when moving between tilings only, have smooth transitions

"""
Observation [0.2, 0.3]

Tilings:
    a0: 0.0  1.0
    a1: 0.5
    a0, a1
        a0: 1.5 2.0
        a1: 0.1 0.5 0.7

Result has size 3 + 2 + 3 * 4 = 17

0 1 0
1 0
0 0 0 0 1 0 0 0 0 0 0 0
"""


class Attribute(Enum):
    GUARD_P1 = lambda o: o["guard"][0]
    GUARD_P2 = lambda o: o["guard"][1]
    MOVE_P1 = lambda o: o["move"][0]
    MOVE_P2 = lambda o: o["move"][1]
    MOVE_PROGRESS_P1 = lambda o: o["move_frame"][0]
    MOVE_PROGRESS_P2 = lambda o: o["move_frame"][1]
    POSITION_P1 = lambda o: o["position"][0]
    POSITION_P2 = lambda o: o["position"][1]


@dataclass
class Tiling:
    breakpoints:    dict[Attribute, np.ndarray[float]]

    @property
    def attributes(self) -> list[Attribute]:
        return list(self.breakpoints)

    @property
    def dimensionality(self) -> int:
        return len(self.breakpoints)

    @property
    def tiles(self) -> int:
        return reduce(int.__mul__, map(self.tiles_of_attribute, self.breakpoints))

    def tiles_of_attribute(self, attribute: Attribute) -> int:
        return len(self.breakpoints[attribute]) + 1


class TileCoding:
    def __init__(
        self,
        tilings: list[Tiling],
    ):
        self.tilings = tilings

    
    def craft_tiling(self, observation: np.ndarray | torch.Tensor, tiling: Tiling) -> torch.Tensor:
        res = torch.zeros((1, tiling.tiles), dtype=torch.bool)
        
        presence = 0
        offset = 1

        for attribute, breakpoints in tiling.breakpoints.items():
            presence += offset * np.sum(breakpoints <= observation[:, attribute.value])
            offset *= tiling.tiles_of_attribute(attribute)
        
        res[0, presence] = True

        return res


    def transform(self, observation: np.ndarray | torch.Tensor) -> torch.Tensor:
        return torch.hstack([
            self.craft_tiling(observation, tiling)
            for tiling in self.tilings
        ])



