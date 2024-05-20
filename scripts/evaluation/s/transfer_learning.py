from main import load_agent, load_agent_parameters
from agents.ql.ql import QFunctionNetwork
from models import to_
from copy import deepcopy
from os import path
from scripts.evaluation.utils import create_eval_env, quick_train_args, quick_agent_args
from scripts.evaluation.custom_loop import WinRateObserver, AgentCustomRun
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.data_collectors import get_data_custom_loop
from gymnasium.spaces import Discrete
from typing import cast
from main import main as main_training
from main import import_agent

BOT = "bot_PT"
BOT_GREEDY = "bot_greedy_PT"
OPPONENT = "curriculum_PT"

def main(seeds: int = 10, timesteps: int = int(1e6), processes: int = 12, y: bool = False):
    result_path = path.splitext(__file__)[0]

    dummy_env, _ = create_eval_env()
    assert dummy_env.observation_space.shape
    assert isinstance(dummy_env.action_space, Discrete)

    if not path.exists(path.join("saved", BOT)):
        print("We don't have a pre-trained bot agent yet, creating one")
        main_training(quick_train_args(
            agent_args=quick_agent_args(BOT, model="to"),
            timesteps=int(1e6),
        ))
    
    if not path.exists(path.join("saved", BOT_GREEDY)):
        print("We don't have a pre-trained bot (greedy) agent yet, creating one")
        main_training(quick_train_args(
            agent_args=quick_agent_args(BOT_GREEDY, model="to", kwargs={"critic_opponent_update": "greedy"}),
            timesteps=int(1e6),
        ))

    bot_agent_parameters = load_agent_parameters(BOT)
    bot_agent, _ = to_(
        observation_space_size=dummy_env.observation_space.shape[0],
        action_space_size=dummy_env.action_space.n,
        **bot_agent_parameters
    )
    load_agent(bot_agent, BOT)

    bot_greedy_agent_parameters = load_agent_parameters(BOT_GREEDY)
    bot_greedy_agent, _ = to_(
        observation_space_size=dummy_env.observation_space.shape[0],
        action_space_size=dummy_env.action_space.n,
        **bot_greedy_agent_parameters
    )
    load_agent(bot_greedy_agent, BOT_GREEDY)

    expected_critic = cast(QFunctionNetwork, bot_agent.a2c.learner.critic)
    greedy_critic = cast(QFunctionNetwork, bot_greedy_agent.a2c.learner.critic)

    agent_control, _ = to_(
        observation_space_size=dummy_env.observation_space.shape[0],
        action_space_size=dummy_env.action_space.n,
    )

    agent_initted_expected = deepcopy(agent_control)
    agent_initted_greedy = deepcopy(agent_control)

    agent_initted_expected_critic = cast(QFunctionNetwork, agent_initted_expected.a2c.learner.critic)
    agent_initted_greedy_critic = cast(QFunctionNetwork, agent_initted_greedy.a2c.learner.critic)
    agent_initted_expected_critic.q_network.load_state_dict(expected_critic.q_network.state_dict())
    agent_initted_greedy_critic.q_network.load_state_dict(greedy_critic.q_network.state_dict())

    opponent_agent_parameters = load_agent_parameters(OPPONENT)
    opponent_agent, _ = import_agent("to", dummy_env, opponent_agent_parameters)
    opponent = opponent_agent.extract_opponent(dummy_env)

    runs = {
        "control": AgentCustomRun(agent_control, opponent.act),
        "initted (expected)": AgentCustomRun(agent_initted_expected, opponent.act),
        "initted (greedy)": AgentCustomRun(agent_initted_greedy, opponent.act),
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
        title="",
        fig_path=result_path,
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "control": "No learned Q-function",
            "initted (expected)": "With learned Q-function (expected)",
            "initted (greedy)": "With learned Q-function (greedy)",
        },
        attr_name="win_rate",
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
