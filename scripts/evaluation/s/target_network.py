from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_train_args

def main(seeds: int = 10, timesteps: int = int(1e6), processes: int = 4):
    runs_raw = {
        "target_network_0": {"critic_target_update_rate": 0},
        "target_network_100": {"critic_target_update_rate": 100},
        "target_network_1000": {"critic_target_update_rate": 1000},
        "target_network_10000": {"critic_target_update_rate": 10000},
    }
    
    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k, model="to", kwargs=v),
        timesteps=timesteps,
    ) for k, v in runs_raw.items()}

    dfs = get_data(
        data="win_rate",
        runs=runs,
        seeds=seeds,
        processes=processes,
    )

    if dfs is None:
        return

    plot_data(
        dfs=dfs,
        title="Win rate over the last 100 episodes against the in-game AI",
        fig_path=path.splitext(__file__)[0],
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "target_network_0":     "Every 0 updates",
            "target_network_100":   "Every 100 updates",
            "target_network_1000":  "Every 1000 updates",
            "target_network_10000": "Every 10000 updates",
        },
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
