from os import path
from models import mimic_
from scripts.evaluation.data_collectors import get_data, get_data_dataset
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_env_args, quick_train_args, create_eval_env
from scripts.evaluation.custom_loop import MimicObserver
from gymnasium.spaces import Discrete
from args import CurriculumArgs

def main(seeds: int = 10, timesteps: int = int(1e6), epochs: int = 10, processes: int = 4, shuffle: bool = True, name_suffix: str = "", y: bool = False):
    result_basename = path.splitext(__file__)[0] + name_suffix

    run_name_mapping = {
        "opp_entropy_coef_004":   "$\\beta = 0.04$",
        "opp_entropy_coef_008":   "$\\beta = 0.08$",
        "opp_entropy_coef_016":   "$\\beta = 0.16$",
        "opp_entropy_coef_032":   "$\\beta = 0.32$",
    }
    
    runs_raw = {
        "opp_entropy_coef_004":   {"opponent_model_entropy_coef": 0.04},
        "opp_entropy_coef_008":   {"opponent_model_entropy_coef": 0.08},
        "opp_entropy_coef_016":   {"opponent_model_entropy_coef": 0.16},
        "opp_entropy_coef_032":   {"opponent_model_entropy_coef": 0.32},
    }

    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k, model="to", kwargs=v),
        timesteps=timesteps,
    ) for k, v in runs_raw.items()}

    # Win rate on normal agent against curriculum
    runs_curriculum = {k: quick_train_args(
        agent_args=quick_agent_args(k, model="to", kwargs=v),
        env_args=quick_env_args(
            curriculum=CurriculumArgs(
                enabled=True,
                episode_threshold=1000,
            ),
        ),
        timesteps=timesteps,
    ) for k, v in runs_raw.items()}
    dfs = get_data(
        data="win_rate",
        runs=runs_curriculum,
        seeds=seeds,
        processes=processes,
        y=y,
    )

    if dfs is None:
        return

    plot_data(
        dfs=dfs,
        title="Win rate over the last 100 episodes against the curriculum",
        fig_path=result_basename + "_wr_curriculum",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping=run_name_mapping,
    )

    # Win rate on normal agent
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
        title="Win rate over the last 100 episodes against the in-game bot",
        fig_path=result_basename + "_wr",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping=run_name_mapping,
    )

    # Losses on normal agent
    dfs = get_data(
        data="learningloss_of_p2s_model",
        runs=runs,
        seeds=seeds,
        processes=processes,
    )

    if dfs is None:
        return

    plot_data(
        dfs=dfs,
        title="Opponent model loss against the in-game bot",
        fig_path=result_basename + "_loss",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Loss",
        run_name_mapping=run_name_mapping
    )

    # Losses on dataset
    runs_dataset_raw = {
        "opp_entropy_coef_004_opp":   {"entropy_coef": 0.04},
        "opp_entropy_coef_008_opp":   {"entropy_coef": 0.08},
        "opp_entropy_coef_016_opp":   {"entropy_coef": 0.16},
        "opp_entropy_coef_032_opp":   {"entropy_coef": 0.32},
    }
    
    dummy_env, _ = create_eval_env()
    assert dummy_env.observation_space.shape
    assert isinstance(dummy_env.action_space, Discrete)

    runs_dataset = {
        k: mimic_(**v)[0] # type: ignore
        for k, v in runs_dataset_raw.items()
    }

    dfs = get_data_dataset(
        result_path=result_basename,
        runs=runs_dataset,
        observer_type=MimicObserver,
        seeds=seeds,
        processes=processes,
        epochs=epochs,
        shuffle=shuffle,
    )

    if dfs is None:
        return
    
    plot_data(
        dfs=dfs,
        title=f"Opponent model loss on the dataset",
        fig_path=f"{result_basename}_loss_dataset",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Loss",
        run_name_mapping={
            "opp_entropy_coef_004_opp":   "$\\beta = 0.04$",
            "opp_entropy_coef_008_opp":   "$\\beta = 0.08$",
            "opp_entropy_coef_016_opp":   "$\\beta = 0.16$",
            "opp_entropy_coef_032_opp":   "$\\beta = 0.32$",
        }
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)