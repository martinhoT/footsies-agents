# %% Make sure we are running in the project's root

from os import chdir
chdir("/home/martinho/projects/footsies-agents")

# %% Imports

import torch
import multiprocessing as mp
from collections import deque
from agents.base import FootsiesAgentBase
from main import load_agent, load_agent_parameters
from agents.ql.ql import QFunctionNetwork
from models import to_
from copy import deepcopy
from functools import partial
from scripts.evaluation.utils import create_env, Observer, test

# %% Prepare environment

dummy_env, _ = create_env()

# %% Import base agent

# Learnt against the in-game bot
AGENT_NAME = "REF"

agent_parameters = load_agent_parameters(AGENT_NAME)
agent, _ = to_(
    observation_space_size=dummy_env.observation_space.shape[0],
    action_space_size=dummy_env.action_space.n,
    **agent_parameters
)
load_agent(agent, AGENT_NAME)

# %% Extract critic

critic: QFunctionNetwork = agent.a2c.learner.critic

# %% Create new agents from scratch, but one of them uses the already-made Q-function network

agent_0, _ = to_(
    observation_space_size=dummy_env.observation_space.shape[0],
    action_space_size=dummy_env.action_space.n,
    **agent_parameters
)

agent_1 = deepcopy(agent_0)

agent_1_critic: QFunctionNetwork = agent_1.a2c.learner.critic
agent_1_critic.q_network.load_state_dict(critic.q_network.state_dict())

# %% Extract opponent

opponent = agent.extract_opponent(dummy_env)

# %% Create an observer class for extracting results from the main loop

class WinRateObserver(Observer):
    def __init__(self, last: int = 100):
        self._wins = deque([], maxlen=last)
        self._win_rates = []

    def update(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict, agent: FootsiesAgentBase):
        if terminated or truncated:
            won = (reward > 0) if terminated else (next_info["guard"][0] > next_info["guard"][1])
            self._wins.append(won)
            self._win_rates.append(sum(self._wins) / len(self._wins))

    @property
    def win_rates(self) -> list[float]:
        return self._win_rates

# %% Test the agents

with mp.Pool(2) as pool:
    test_1000 = partial(test, episodes=10000)
    observers: list[WinRateObserver] = pool.starmap(test_1000, (
        (agent_0, "control", 0, WinRateObserver, opponent.act),
        (agent_1, "initted", 1, WinRateObserver, opponent.act),
    ))

# %% Plot results

from os import path
import matplotlib.pyplot as plt

result_path, _ = path.splitext(__file__)

plt.plot(observers[0].win_rates)
plt.plot(observers[1].win_rates)
plt.legend(["No learned Q-function", "With learned Q-function"])
plt.title("Adaptation performance between agent\nwith a pre-trained Q-function vs one with,\nin terms of win rate over the last 100 episodes")
plt.xlabel("Episode")
plt.ylabel("Win rate")
plt.savefig(result_path)

# %% Save the data for posterity

import numpy as np

data = np.array([observers[0].win_rates, observers[1].win_rates])
np.save(result_path, data)
