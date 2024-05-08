import torch
from agents.base import FootsiesAgentBase
from main import load_agent_parameters
from agents.ql.ql import QFunctionNetwork
from models import to_
from os import path
from scripts.evaluation.utils import create_eval_env
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.data_collectors import AgentCustomRun, get_data_custom_loop
from scripts.evaluation.custom_loop import Observer
from collections import deque
from functools import partial
from typing import cast
from gymnasium.spaces import Discrete

class QSumObserver(Observer):
    def __init__(self, critic: QFunctionNetwork, last: int = 100):
        self._last_sums = deque([], maxlen=last)
        self._idxs: list[int] = []
        self._sums: list[float] = []
        self._current_step = 0
        self._critic = critic

    def update(self, step: int, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict, agent: FootsiesAgentBase):
        if self._current_step > 0 and (self._current_step % self._last_sums.maxlen) == 0: # type: ignore
            avg_sum = sum(self._last_sums) / len(self._last_sums)
            self._idxs.append(step)
            self._sums.append(avg_sum)
        
        qs = cast(torch.Tensor, self._critic.q(obs))
        self._last_sums.append(qs.sum())
        self._current_step = (self._current_step + 1) % self._last_sums.maxlen # type: ignore
        
    @property
    def data(self) -> tuple[list[int], tuple[list[float], ...]]:
        return self._idxs, (self._sums,)

    @staticmethod
    def attributes() -> tuple[str, ...]:
        return ("q_sum",)


def main(seeds: int = 10, timesteps: int = int(1e6), processes: int = 12, y: bool = False):
    env, _ = create_eval_env()
    assert env.observation_space.shape
    assert isinstance(env.action_space, Discrete)

    # Learnt against the in-game AI
    agent_name = "REF"
    agent_parameters = load_agent_parameters(agent_name)
    agent, _ = to_(
        observation_space_size=env.observation_space.shape[0],
        action_space_size=env.action_space.n,
        **agent_parameters
    )
    agent.load(f"saved/{agent_name}")

    critic = cast(QFunctionNetwork, agent.a2c.learner.critic)
    agent.learn_opponent_model = False
    agent.learn_game_model = False
    agent.learn_a2c = True
    agent.a2c.learner.learn_actor = False

    opponent = agent.extract_opponent(env)

    observer = cast(type[QSumObserver], partial(QSumObserver, critic=critic))

    agents = {}
    for name, critic_agent_update, critic_opponent_update in (
        ("ql_es", "q_learning", "expected_sarsa"),
        ("ql_ql", "q_learning", "q_learning"),
        ("es_ql", "expected_sarsa", "expected_sarsa"),
        ("es_es", "expected_sarsa", "expected_sarsa"),
    ):
        agent_parameters = load_agent_parameters(agent_name)
        agent_parameters["critic_agent_update"] = critic_agent_update
        agent_parameters["critic_opponent_update"] = critic_opponent_update
        agent, _ = to_(
            observation_space_size=env.observation_space.shape[0],
            action_space_size=env.action_space.n,
            **agent_parameters
        )
        agent.load(f"saved/{agent_name}")

        agents[name] = AgentCustomRun(agent, opponent.act)

    result_path = path.splitext(__file__)[0] 

    dfs = get_data_custom_loop(result_path, agents, observer,
        seeds=seeds,
        timesteps=timesteps,
        processes=processes,
        y=y,
    )

    if dfs is None:
        return

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

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
