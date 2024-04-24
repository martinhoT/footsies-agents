from os import path
from scripts.evaluation.utils import get_and_plot_data

def main(seeds: int):
    agents = [
        ("sparse_reward_curriculum", {}, {"dense_reward": False}, {"curriculum": True, "curriculum_threshold": 1000}),
        ("dense_reward_curriculum", {}, {"dense_reward": True}, {"curriculum": True, "curriculum_threshold": 1000}),
    ]

    get_and_plot_data(
        data="performancewin_rate_against_current_curriculum_opponent",
        agents=agents,
        title="Win rate over the last 100 episodes against the curriculum",
        fig_path=path.splitext(__file__)[0],
        seeds=seeds,
        timesteps=None,
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "sparse_reward_curriculum":    "Sparse reward",
            "dense_reward_curriculum":     "Dense reward",
        }
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)