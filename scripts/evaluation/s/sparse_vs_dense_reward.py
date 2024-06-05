from os import path
from scripts.evaluation.data_collectors import get_data, get_data_custom_loop
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_env_args, quick_train_args, create_agent_the_one
from scripts.evaluation.custom_loop import AgentCustomRun, WinRateObserver
from opponents.curriculum import WhiffPunisher

def main(seeds: int | None = None, timesteps: int = int(2e6), processes: int = 12, y: bool = False):
    if seeds is None:
        seeds = 10
    
    result_path = path.splitext(__file__)[0]

    runs_raw = {
        "sparse_reward": {"dense_reward": False},
        "dense_reward": {"dense_reward": True},
    }

    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k),
        env_args=quick_env_args(kwargs=v),
        timesteps=timesteps,
    ) for k, v in runs_raw.items()}

    dfs = get_data(
        data="win_rate",
        runs=runs,
        seeds=seeds,
        processes=processes,
        y=y,
    )

    if dfs is None:
        return

    plot_data(
        dfs=dfs,
        title="",
        fig_path=result_path + "_wr",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "sparse_reward":    "Sparse reward",
            "dense_reward":     "Dense reward",
        },
        ylim=(0, 1),
    )

    # Against WhiffPunisher

    runs_whiff_raw = {
        "sparse_reward_whiff": {"dense_reward": False},
        "dense_reward_whiff": {"dense_reward": True},
    }

    whiff_punisher = WhiffPunisher()

    runs_whiff = {k: AgentCustomRun(
        agent=create_agent_the_one,
        opponent=whiff_punisher.act,
        env_args=quick_env_args(kwargs=v),
    ) for k, v in runs_whiff_raw.items()}

    dfs = get_data_custom_loop(
        result_path=result_path + "_whiff",
        runs=runs_whiff,
        observer_type=WinRateObserver,
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
        fig_path=result_path + "_whiff_wr",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "sparse_reward_whiff":  "Sparse reward",
            "dense_reward_whiff":   "Dense reward"
        },
        attr_name="win_rate",
        ylim=(0, 1),
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
