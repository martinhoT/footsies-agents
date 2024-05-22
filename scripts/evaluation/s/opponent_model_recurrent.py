from os import path
from models import mimic_
from scripts.evaluation.data_collectors import get_data, get_data_dataset
from scripts.evaluation.plotting import plot_data
from scripts.evaluation.utils import quick_agent_args, quick_train_args, create_eval_env
from scripts.evaluation.custom_loop import MimicObserver
from gymnasium.spaces import Discrete

def main(seeds: int | None = None, timesteps: int = int(1e6), epochs: int = 5, processes: int = 12, shuffle: bool = True, name_suffix: str = "", y: bool = False):
    if seeds is None:
        seeds = 3

    result_basename = path.splitext(__file__)[0] + name_suffix

    run_name_mapping = {
        "opp_normal":               "Non-recurrent",
        "opp_recurrent_end":        "Recurrent (end)",
        "opp_recurrent_hit":        "Recurrent (hit)",
        "opp_recurrent_neutral":    "Recurrent (neutral)"
    }
    
    runs_raw = {
        "opp_normal": {"opponent_model_recurrent": False},
        "opp_recurrent_end": {"opponent_model_recurrent": True, "opponent_model_reset_context_at": "end"},
        "opp_recurrent_hit": {"opponent_model_recurrent": True, "opponent_model_reset_context_at": "hit"},
        "opp_recurrent_neutral": {"opponent_model_recurrent": True, "opponent_model_reset_context_at": "neutral"},
    }

    runs = {k: quick_train_args(
        agent_args=quick_agent_args(k, model="to", kwargs=v),
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

    # Sort the values, there seemed to be some issue with S0
    for df in dfs.values():
        df.sort_values("Idx", inplace=True, ascending=True)

    plot_data(
        dfs=dfs,
        title="",
        fig_path=result_basename + "_loss",
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Loss",
        run_name_mapping=run_name_mapping
    )

    # Losses on dataset
    runs_dataset_raw = {
        "dataset_opp_normal": {"recurrent": False},
        "dataset_opp_recurrent_end": {"recurrent": True, "reset_context_at": "end"},
        "dataset_opp_recurrent_hit": {"recurrent": True, "reset_context_at": "hit"},
        "dataset_opp_recurrent_neutral": {"recurrent": True, "reset_context_at": "neutral"},
    }
    
    dummy_env, _ = create_eval_env()
    assert dummy_env.observation_space.shape
    assert isinstance(dummy_env.action_space, Discrete)

    runs_dataset = {
        k: mimic_(
            observation_space_size=dummy_env.observation_space.shape[0],
            action_space_size=int(dummy_env.action_space.n),
            **v,
        )[0] for k, v in runs_dataset_raw.items()
    }

    dfs = get_data_dataset(
        result_path=result_basename,
        runs=runs_dataset,
        observer_type=MimicObserver,
        seeds=seeds,
        processes=min(processes, 10), # the PC gets way too hot if all CPUs are constantly running
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
            xlabel="Iteration",
            ylabel="Loss",
            run_name_mapping={"dataset_" + k: v for k, v in run_name_mapping.items()},
            attr_name=f"{player}_loss",
        )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)