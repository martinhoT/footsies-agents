from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_train_args

def main(seeds: int | None = None, timesteps: int = int(1e6), processes: int = 12, y: bool = False):
    raise RuntimeError("deprecated test")
    
    if seeds is None:
        seeds = 3

    runs_raw = {
        "target_network_0": 0,
        "target_network_100": 100,
        "target_network_1000": 1000,
        "target_network_10000": 10000,
    }
    
    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k, model="to", kwargs={
            "critic_agent_update": "q_learning", # dramatize it
            "critic_opponent_update": "q_learning", # dramatize it
            "critic_target_update_rate": update_rate,
        }),
        timesteps=timesteps,
    ) for k, update_rate in runs_raw.items()}

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
