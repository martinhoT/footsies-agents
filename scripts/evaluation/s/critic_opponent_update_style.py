from os import path
from scripts.evaluation.data_collectors import get_data
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_train_args

def main(seeds: int | None = None, timesteps: int = int(1e6), processes: int = 12, y: bool = False):
    if seeds is None:
        seeds = 6
    
    runs_raw = {
        "critic_opponent_style_sarsa": {"critic_opponent_update": "sarsa"},
        "critic_opponent_style_expected_sarsa": {"critic_opponent_update": "expected_sarsa"},
        "critic_opponent_style_q_learning": {"critic_opponent_update": "q_learning"},
        "critic_opponent_style_uniform": {"critic_opponent_update": "uniform"},
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
            "critic_opponent_style_sarsa":          "Sarsa",
            "critic_opponent_style_expected_sarsa": "Expected Sarsa",
            "critic_opponent_style_q_learning":     "Greedy",
            "critic_opponent_style_uniform":        "Uniform",
        },
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
