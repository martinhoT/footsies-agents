from os import path
from models import mimic_
from scripts.evaluation.data_collectors import get_data, get_data_dataset
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_env_args, quick_train_args, create_eval_env
from scripts.evaluation.custom_loop import MimicObserver
from gymnasium.spaces import Discrete

def main(seeds: int | None = None, timesteps: int = int(1e6), epochs: int = 5, processes: int = 12, shuffle: bool = True, name_suffix: str = "", y: bool = False, do_dataset: bool = False):
    if seeds is None:
        seeds = 3
    
    result_basename = path.splitext(__file__)[0] + name_suffix

    run_name_mapping = {
        "opp_no_dynamic_weights":   "No dynamic weights",
        "opp_yes_dynamic_weights":  "Dynamic weights",
    }
    
    runs_raw = {
        "opp_no_dynamic_weights": {"opponent_model_dynamic_loss_weights": False},
        "opp_yes_dynamic_weights": {"opponent_model_dynamic_loss_weights": True},
    }

    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k, model="to", kwargs=v),
        env_args=quick_env_args(kwargs={"dense_reward": True}),
        timesteps=timesteps,
    ) for k, v in runs_raw.items()}


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
        title="",
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
        title="",
        fig_path=result_basename + "_loss",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Loss",
        run_name_mapping=run_name_mapping
    )

    # Losses on normal agent
    dfs = get_data(
        data="learningprediction_score_of_p2s_model",
        runs=runs,
        seeds=seeds,
        processes=processes,
    )

    if dfs is None:
        return

    plot_data(
        dfs=dfs,
        title="",
        fig_path=result_basename + "_score",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Score",
        run_name_mapping=run_name_mapping
    )

    # Losses on dataset
    if do_dataset:
        runs_dataset_raw = {
            "dataset_opp_no_dynamic_weights": {"dynamic_loss_weights": False},
            "dataset_opp_yes_dynamic_weights": {"dynamic_loss_weights": True},
        }
        
        dummy_env, _ = create_eval_env()
        assert dummy_env.observation_space.shape
        assert isinstance(dummy_env.action_space, Discrete)

        runs_dataset = {
            k: mimic_(
                observation_space_size=dummy_env.observation_space.shape[0],
                action_space_size=int(dummy_env.action_space.n),
                **v, # type: ignore
            )[0] 
            for k, v in runs_dataset_raw.items()
        }

        dfs = get_data_dataset(
            result_path=result_basename,
            runs=runs_dataset,
            observer_type=MimicObserver,
            seeds=seeds,
            processes=min(6, processes), # the PC gets way too hot if all CPUs are constantly running
            epochs=epochs,
            shuffle=shuffle,
            y=y,
        )

        if dfs is None:
            return
        
        for player in ("p1", "p2"):
            plot_data(
                dfs=dfs,
                title=f"Opponent model loss on the dataset ({player.upper()})",
                fig_path=f"{result_basename}_loss_dataset_{player}",
                exp_factor=0.9,
                xlabel="Time step",
                ylabel="Loss",
                run_name_mapping={"dataset_" + k: v for k, v in run_name_mapping.items()},
                attr_name=f"{player}_loss",
            )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
