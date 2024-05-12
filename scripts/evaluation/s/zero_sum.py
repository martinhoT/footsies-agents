import torch
from main import load_agent_parameters
from models import to_
from os import path
from scripts.evaluation.utils import create_eval_env
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.data_collectors import get_data_custom_loop
from scripts.evaluation.custom_loop import Observer, AgentCustomRun
from collections import deque
from gymnasium.spaces import Discrete
from agents.the_one.agent import TheOneAgent
from copy import deepcopy

class QSumObserver(Observer):
    def __init__(self, last: int = 100):
        self._last_sums: deque[float] = deque([], maxlen=last)
        self._idxs: list[int] = []
        self._sums: list[float] = []

    def update(self, step: int, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict, agent: TheOneAgent):
        if step > 0 and (step % self._last_sums.maxlen) == 0: # type: ignore
            avg_sum = sum(self._last_sums) / len(self._last_sums)
            self._idxs.append(step)
            self._sums.append(avg_sum)
        
        qs = agent.a2c.learner.critic.q(obs)
        self._last_sums.append(qs.sum().item())
        
    @property
    def data(self) -> tuple[list[int], tuple[list[float], ...]]:
        return self._idxs, (self._sums,)

    @staticmethod
    def attributes() -> tuple[str, ...]:
        return ("q_sum",)


def main(seeds: int = 10, timesteps: int = int(1e6), processes: int = 12, y: bool = False):
    raise RuntimeError("deprecated test")
    
    env, _ = create_eval_env()
    assert env.observation_space.shape
    assert isinstance(env.action_space, Discrete)

    # Learnt against the in-game AI
    agent_name = "REF"
    agent_parameters = load_agent_parameters(agent_name)
    agent, _ = to_(
        observation_space_size=env.observation_space.shape[0],
        action_space_size=int(env.action_space.n),
        **agent_parameters
    )
    agent.load(f"saved/{agent_name}")

    agent.learn_opponent_model = False
    agent.learn_game_model = False
    agent.learn_a2c = True
    agent.a2c.learner.learn_actor = False

    runs = {}
    for name, critic_agent_update, critic_opponent_update in (
        ("ql_es", "q_learning", "expected_sarsa"),
        ("ql_ql", "q_learning", "q_learning"),
        ("es_ql", "expected_sarsa", "q_learning"),
        ("es_es", "expected_sarsa", "expected_sarsa"),
    ):
        new_agent = deepcopy(agent)
        new_agent.a2c.learner.agent_update_style = getattr(new_agent.a2c.learner.UpdateStyle, critic_agent_update.upper())
        new_agent.a2c.learner.opponent_update_style = getattr(new_agent.a2c.learner.UpdateStyle, critic_opponent_update.upper())

        runs[name] = AgentCustomRun(new_agent, None)

    result_path = path.splitext(__file__)[0] 

    dfs = get_data_custom_loop(result_path, runs, QSumObserver,
        seeds=seeds,
        timesteps=timesteps,
        processes=processes,
        y=y,
    )

    if dfs is None:
        return

    plot_data(
        dfs=dfs,
        title="",
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
        attr_name="q_sum",
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
