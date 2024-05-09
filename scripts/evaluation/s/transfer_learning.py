from main import load_agent, load_agent_parameters
from agents.ql.ql import QFunctionNetwork
from models import to_
from copy import deepcopy
from os import path
from scripts.evaluation.utils import create_eval_env
from scripts.evaluation.custom_loop import WinRateObserver, AgentCustomRun
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.data_collectors import get_data_custom_loop
from gymnasium.spaces import Discrete
from typing import cast

def main(seeds: int = 10, timesteps: int = int(1e6), processes: int = 12, y: bool = False):
    result_path = path.splitext(__file__)[0]

    dummy_env, _ = create_eval_env()
    assert dummy_env.observation_space.shape
    assert isinstance(dummy_env.action_space, Discrete)

    # Learnt against the in-game AI
    AGENT_NAME = "REF"

    agent_parameters = load_agent_parameters(AGENT_NAME)
    agent, _ = to_(
        observation_space_size=dummy_env.observation_space.shape[0],
        action_space_size=dummy_env.action_space.n,
        **agent_parameters
    )
    load_agent(agent, AGENT_NAME)

    critic = cast(QFunctionNetwork, agent.a2c.learner.critic)

    agent_0, _ = to_(
        observation_space_size=dummy_env.observation_space.shape[0],
        action_space_size=dummy_env.action_space.n,
        **agent_parameters
    )

    agent_1 = deepcopy(agent_0)

    agent_1_critic = cast(QFunctionNetwork, agent_1.a2c.learner.critic)
    agent_1_critic.q_network.load_state_dict(critic.q_network.state_dict())

    opponent = agent.extract_opponent(dummy_env)

    runs = {
        "control": AgentCustomRun(agent_0, opponent.act),
        "initted": AgentCustomRun(agent_1, opponent.act),
    }

    dfs = get_data_custom_loop(result_path, runs, WinRateObserver, 
        seeds=seeds,
        timesteps=timesteps,
        processes=processes,
        y=y,
    )
    if dfs is None:
        return

    plot_data(
        dfs=dfs,
        title="Win rate over the last 100 episodes",
        fig_path=result_path,
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "control": "No learned Q-function",
            "initted": "With learned Q-function",
        },
        attr_name="win_rate",
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
