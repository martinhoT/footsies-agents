from os import path
from scripts.evaluation.utils import create_eval_env, quick_agent_args, quick_train_args, quick_env_args
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.data_collectors import get_data
from gymnasium.spaces import Discrete
from args import SelfPlayArgs

def main(seeds: int = 10, timesteps: int = int(3e6), processes: int = 12, y: bool = False):
    result_path = path.splitext(__file__)[0]

    dummy_env, _ = create_eval_env()
    assert dummy_env.observation_space.shape
    assert isinstance(dummy_env.action_space, Discrete)

    runs = {
        "self_play": quick_train_args(
            agent_args=quick_agent_args("self_play", model="to"),
            env_args=quick_env_args(
                self_play=SelfPlayArgs(
                    enabled=True,
                    max_opponents=10,
                    snapshot_interval=2000,
                    switch_interval=100,
                    mix_bot=1,
                    add_curriculum_opps=False,
                    evaluate_every=100,
                ),
                wrapper_time_limit=1000,
            ),
            timesteps=timesteps,
        )
    }

    dfs = get_data(
        data="performanceelo",
        runs=runs,
        seeds=seeds,
        processes=processes,
        y=y,
    )

    if dfs is None:
        return

    plot_data(
        dfs=dfs,
        title="Elo during self-play",
        fig_path=result_path + "_elo",
        exp_factor=0.99,
        xlabel="Episode",
        ylabel="Elo",
        run_name_mapping=None,
    )

    dfs = get_data(
        data="episode_length",
        runs=runs,
        seeds=seeds,
        processes=processes,
        y=y,
    )

    if dfs is None:
        return
    
    plot_data(
        dfs=dfs,
        title="Episode length during self-play",
        fig_path=result_path + "_length",
        exp_factor=0.99,
        xlabel="Episode",
        ylabel="Time steps",
        run_name_mapping=None,
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
