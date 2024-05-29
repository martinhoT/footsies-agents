import os
from os import path
from scripts.evaluation.utils import quick_agent_args, quick_train_args, quick_env_args
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.data_collectors import get_data
from args import SelfPlayArgs, CurriculumArgs
from main import main as main_training

def perform_run(
    dense_reward: bool,
    timesteps: int,
    seeds: int,
    processes: int,
    y: bool,
    result_path: str,
):
    suffix = "_dense" if dense_reward else "_sparse"
    result_path = result_path + suffix

    runs = {
        "self_play" + suffix: quick_train_args(
            agent_args=quick_agent_args("self_play" + suffix, model="to"),
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
                kwargs={
                    "dense_reward": dense_reward,
                }
            ),
            timesteps=timesteps,
        )
    }

    # Elo

    dfs = get_data(
        data="performanceelo",
        runs=runs,
        seeds=seeds,
        processes=processes,
        y=y,
        pre_trained="curriculum_PT",
    )

    if dfs is None:
        return

    plot_data(
        dfs=dfs,
        title="",
        fig_path=result_path + "_elo",
        exp_factor=0.99,
        xlabel="Episode",
        ylabel="Elo",
        run_name_mapping=None,
    )

    # Episode length

    dfs = get_data(
        data="episode_length",
        runs=runs,
        seeds=seeds,
        processes=processes,
        y=y,
        pre_trained="curriculum_PT",
    )

    if dfs is None:
        return
    
    plot_data(
        dfs=dfs,
        title="",
        fig_path=result_path + "_length",
        exp_factor=0.99,
        xlabel="Episode",
        ylabel="Time steps",
        run_name_mapping=None,
        ylim=(0, None),
    )

    # Episode truncations

    dfs = get_data(
        data="episode_truncations",
        runs=runs,
        seeds=seeds,
        processes=processes,
        y=y,
        pre_trained="curriculum_PT",
    )

    if dfs is None:
        return
    
    plot_data(
        dfs=dfs,
        title="",
        fig_path=result_path + "_truncations",
        exp_factor=0.99,
        xlabel="Episode",
        ylabel="Truncation probability",
        run_name_mapping=None,
    )

def main(seeds: int | None = None, timesteps: int = int(3e6), processes: int = 12, y: bool = False, just_pre_train: bool = False, do_sparse: bool = False):
    if seeds is None:
        seeds = 10
    
    result_path = path.splitext(__file__)[0]

    # Just train against the curriculum once, and load that agent
    if not path.exists("saved/curriculum_PT"):
        print("We don't have a pre-trained curriculum agent yet, creating one")
        main_training(quick_train_args(
            agent_args=quick_agent_args("curriculum_PT", model="to"),
            env_args=quick_env_args(
                curriculum=CurriculumArgs(True),
                kwargs={"dense_reward": True},
            ),
            timesteps=None,
        ))
    
    else:
        print("We already have a pre-trained agent, continuing")

    if just_pre_train:
        return

    perform_run(
        dense_reward=True,
        timesteps=timesteps,
        seeds=seeds,
        processes=processes,
        y=y,
        result_path=result_path,
    )

    if do_sparse:
        perform_run(
            dense_reward=False,
            timesteps=timesteps,
            seeds=seeds,
            processes=processes,
            y=y,
            result_path=result_path,
        )  

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
