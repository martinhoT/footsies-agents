# %% Make sure we are running in the project's root

from os import chdir
chdir("/home/martinho/projects/footsies-agents")

# %% Imports

import torch
from agents.base import FootsiesAgentBase
from main import load_agent, load_agent_parameters
from agents.ql.ql import QFunctionNetwork
from models import to_
from copy import deepcopy
from os import path
from scripts.evaluation.utils import create_env, plot_data, get_data_custom_loop, Observer
from collections import deque
from agents.action import ActionMap
from functools import partial

# %% Define custom observer to check Q-value matrix sum

class QSumObserver(Observer):
    def __init__(self, critic: QFunctionNetwork, last: int = 100):
        self._last_sums = deque([], maxlen=last)
        self._sums = []
        self._current_step = 0
        self._critic = critic

    def update(self, step: int, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict, agent: FootsiesAgentBase):
        if self._current_step > 0 and (self._current_step % self._last_sums.maxlen) == 0:
            avg_sum = sum(self._last_sums) / len(self._last_sums)
            self._sums.append((step, avg_sum))
        
        self._last_sums.append(self._critic.q(obs).sum())
        self._current_step = (self._current_step + 1) % self._last_sums.maxlen
        
    @property
    def data(self) -> list[tuple[int, float]]:
        return self._sums

# %% Prepare environment

env, _ = create_env()

# %% Define the agents

# Learnt against the in-game bot
AGENT_NAME = "REF"

agents = {}
for name, critic_agent_update, critic_opponent_update in (
    ("ql_es", "q_learning", "expected_sarsa"),
    ("ql_ql", "q_learning", "q_learning"),
    ("es_ql", "expected_sarsa", "expected_sarsa"),
    ("es_es", "expected_sarsa", "expected_sarsa"),
):
    agent_parameters = load_agent_parameters(AGENT_NAME)
    agent_parameters["critic_agent_update"] = critic_agent_update
    agent_parameters["critic_opponent_update"] = critic_opponent_update
    agent, _ = to_(
        observation_space_size=env.observation_space.shape[0],
        action_space_size=env.action_space.n,
        **agent_parameters
    )
    agent.load(f"saved/{AGENT_NAME}")

    agents[name] = agent

# %% Extract critic

critic: QFunctionNetwork = agent.a2c.learner.critic
agent.learn_opponent_model = False
agent.learn_game_model = False
agent.learn_a2c = True
agent.a2c.learner.learn_actor = False

# %% Instantiate observer

opponent = agent.extract_opponent(env)

observer = partial(QSumObserver, critic=critic)

# %% Get data, if possible

result_path = path.splitext(__file__)[0] 

dfs = get_data_custom_loop(result_path, agents, observer, opponent, seeds=10, timesteps=1000000)
if dfs is None:
    print("Could not get data, quitting")
    exit(0)

# %% Plot results

plot_data(
    dfs=dfs,
    title="Average Q-matrix sum over the last 100 episodes, for different update styles (Agent | Opponent)",
    fig_path=result_path,
    exp_factor=0.9,
    xlabel="Time step",
    ylabel="Sum",
    run_name_mapping={
        "ql_es": "Greedy | Expected",
        "ql_ql": "Greedy | Greedy",
        "es_ql": "Expected | Greedy",
        "es_es": "Expected | Expected",
    },
)
