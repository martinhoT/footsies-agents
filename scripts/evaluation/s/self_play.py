from main import load_agent_parameters
from models import to_
from os import path
from scripts.evaluation.utils import create_env
from scripts.evaluation.custom_loop import WinRateObserver
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.data_collectors import get_data_custom_loop, AgentCustomRun
from gymnasium.spaces import Discrete

def main(seeds: int = 10, timesteps: int = int(1e6), processes: int = 4):
    result_path = path.splitext(__file__)[0]

    dummy_env, _ = create_env()
    assert dummy_env.observation_space.shape
    assert isinstance(dummy_env.action_space, Discrete)

    # Learnt using self-play
    AGENT_NAME = "sf"

    agent_parameters = load_agent_parameters(AGENT_NAME)
    agent, _ = to_(
        observation_space_size=dummy_env.observation_space.shape[0],
        action_space_size=dummy_env.action_space.n,
        **agent_parameters
    )
    agent.load(f"saved/{AGENT_NAME}")

    runs = {
        "main": AgentCustomRun(agent, None)
    }

    dfs = get_data_custom_loop(
        result_path=result_path,
        runs=runs,
        observer_type=WinRateObserver,
        seeds=seeds,
        timesteps=timesteps,
        processes=processes,
    )

    if dfs is None:
        return

    plot_data(
        dfs=dfs,
        title="Win rate over the last 100 episodes against the in-game AI",
        fig_path=result_path,
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping=None,
        attr_name="win_rate",
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
