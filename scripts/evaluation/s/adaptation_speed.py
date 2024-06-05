import random
import torch as T
import matplotlib.pyplot as plt
from os import path
from scripts.evaluation.utils import create_eval_env
from scripts.evaluation.data_collectors import get_data_custom_loop
from scripts.evaluation.custom_loop import WinRateObserver, PreCustomLoop, AgentCustomRun
from models import to_
from copy import deepcopy
from opponents.curriculum import Idle, Backer, NSpammer, BSpammer, NSpecialSpammer, BSpecialSpammer, WhiffPunisher, CurriculumOpponent
from typing import cast, Callable, Literal
from agents.the_one.agent import TheOneAgent
from dataclasses import dataclass
from opponents.base import Opponent
from tqdm import trange
from agents.action import ActionMap
from gymnasium.spaces import Discrete
from functools import partial
from gymnasium import Env
from footsies_gym.envs.footsies import FootsiesEnv
import seaborn as sns


OPPONENT_MAP: dict[str, Callable[[], CurriculumOpponent | None]] = {
    "Idle": Idle,
    "Backer": Backer,
    "NSpammer": NSpammer,
    "BSpammer": BSpammer,
    "NSpecialSpammer": NSpecialSpammer,
    "BSpecialSpammer": BSpecialSpammer,
    "WhiffPunisher": WhiffPunisher,
    "Bot": lambda: None,
}

@dataclass
class AgentTrainingRegime:
    name:                   str
    agent:                  TheOneAgent
    training_opponent:      Opponent | None | Literal["Blank"]
    evaluation_opponent:    Opponent | None


def train_agent_against_opponent(agent: TheOneAgent, env: Env, footsies_env: FootsiesEnv, initial_seed: int | None, opponent: Opponent | None, timesteps: int = int(1e6), label: str = ""):
    T.manual_seed(initial_seed)
    random.seed(initial_seed)
    
    o = opponent.act if opponent is not None else None
    footsies_env.set_opponent(o)

    seed = initial_seed
    terminated, truncated = True, True
    for _ in trange(timesteps, desc=label, unit="step", dynamic_ncols=True, colour="#FF9E64"):
        if (terminated or truncated):
            obs, info = env.reset(seed=seed)
            terminated, truncated = False, False
            seed = None

        action = agent.act(obs, info)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        reward = float(reward)

        # Skip hitstop freeze
        in_hitstop = ActionMap.is_in_hitstop_ori(next_info, True) or ActionMap.is_in_hitstop_ori(next_info, False)
        while in_hitstop and obs.isclose(next_obs).all():
            action = agent.act(obs, info)
            next_obs, r, terminated, truncated, next_info = env.step(action)
            r = float(r)
            reward += r

        agent.update(obs, next_obs, reward, terminated, truncated, info, next_info)

        obs, info = next_obs, next_info


def heatplot(mtx: T.Tensor, xlabels: list[str], ylabels: list[str], fig_path: str | None = None, color_reverse: bool = False):
    ax = sns.heatmap(mtx,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis" + ("_r" if color_reverse else ""), # "crest" or "viridis" or "mako"
        annot=True,
        fmt=".2f",
        cbar=True,
        square=True,
        xticklabels=xlabels,
        yticklabels=ylabels,
    ) 
    ax.set_xlabel("Evaluation opponent")
    ax.set_ylabel("Training opponent")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    
    fig = ax.get_figure()
    assert fig is not None
    fig.set_size_inches(5.5, 5.5)
    fig.tight_layout()
    if fig_path is not None:
        fig.savefig(fig_path)
    fig.clear()
    plt.close(fig)


def main(
    seeds: int | None = None,
    timesteps: int = int(1e6),
    processes: int = 12,
    y: bool = False,
    small: bool = False,
    wr_thresh: float = 0.8,
):
    if seeds is None:
        seeds = 6

    result_path = path.splitext(__file__)[0]

    dummy_env, _  = create_eval_env()
    assert dummy_env.observation_space.shape
    assert isinstance(dummy_env.action_space, Discrete)

    agent, _ = to_(
        observation_space_size=dummy_env.observation_space.shape[0],
        action_space_size=int(dummy_env.action_space.n),
    )

    if small:
        opponent_labels = ["Blank", "NSpammer", "BSpammer"]
    else:
        opponent_labels = ["Blank", "NSpammer", "BSpammer", "NSpecialSpammer", "BSpecialSpammer", "WhiffPunisher", "Bot"]

    # The same as the opponent labels list but the "Bot" label is substituted with "In-game AI" for consistency with the thesis doc
    opponent_labels_consistent = (opponent_labels[:-1] + ["In-game AI"]) if not small else opponent_labels

    agent_regimes: list[AgentTrainingRegime] = []
    for training_opponent_label in opponent_labels:
        for evaluation_opponent_label in opponent_labels:
            if evaluation_opponent_label == "Blank":
                continue
            
            if training_opponent_label == "Blank":
                training_opponent = "Blank"
            else:
                training_opponent = OPPONENT_MAP[training_opponent_label]()
            
            evaluation_opponent = OPPONENT_MAP[evaluation_opponent_label]()

            agent_regime = AgentTrainingRegime(
                name=f"{training_opponent_label}_to_{evaluation_opponent_label}",
                agent=deepcopy(agent),
                training_opponent=training_opponent,
                evaluation_opponent=evaluation_opponent,
            )

            agent_regimes.append(agent_regime)

    runs = {
        reg.name: AgentCustomRun(
            agent=reg.agent,
            opponent=reg.evaluation_opponent.act if reg.evaluation_opponent is not None else reg.evaluation_opponent,
            pre_loop=cast(PreCustomLoop, partial(train_agent_against_opponent, opponent=reg.training_opponent, timesteps=timesteps, label=reg.name)) if reg.training_opponent != "Blank" else None,
        ) for reg in agent_regimes
    }

    dfs = get_data_custom_loop(
        result_path=result_path,
        runs=runs,
        observer_type=WinRateObserver,
        seeds=seeds,
        timesteps=timesteps,
        processes=processes,
        y=y,
    )

    if dfs is None:
        return

    # Time taken to adapt (get to wr_thresh)

    # Create the matrix first. We exclude "Blank" as an evaluation opponent
    mtx = T.zeros((len(opponent_labels), len(opponent_labels) - 1))
    for name, df in dfs.items():
        training_opponent_label, evaluation_opponent_label = name.split("_to_")
        row = opponent_labels.index(training_opponent_label)
        col = opponent_labels.index(evaluation_opponent_label) - 1
        
        time_taken = 0
        for seed in range(seeds):
            d = df[(df[f"win_rate{seed}"] > wr_thresh) & (df["Idx"] >= int(1e5))]
            l = d.iloc[0, :]["Idx"] if len(d) > 0 else timesteps
            time_taken += l
        time_taken /= seeds

        mtx[row, col] = time_taken / timesteps

    # Plot the heatmap
    heatplot(
        mtx=mtx,
        fig_path=result_path + "_time",
        xlabels=opponent_labels_consistent[1:],
        ylabels=opponent_labels_consistent,
        color_reverse=True,
    )

    # Final win rate

    mtx = T.zeros((len(opponent_labels), len(opponent_labels) - 1))
    for name, df in dfs.items():
        training_opponent_label, evaluation_opponent_label = name.split("_to_")
        row = opponent_labels.index(training_opponent_label)
        col = opponent_labels.index(evaluation_opponent_label) - 1

        mtx[row, col] = df.iloc[-1, :]["win_rateMean"]

    # Plot the heatmap
    heatplot(
        mtx=mtx,
        fig_path=result_path + "_wr_final",
        xlabels=opponent_labels_consistent[1:],
        ylabels=opponent_labels_consistent,
    )

    # Maximum win rate

    mtx = T.zeros((len(opponent_labels), len(opponent_labels) - 1))
    for name, df in dfs.items():
        training_opponent_label, evaluation_opponent_label = name.split("_to_")
        row = opponent_labels.index(training_opponent_label)
        col = opponent_labels.index(evaluation_opponent_label) - 1

        mtx[row, col] = df["win_rateMean"].max()

    # Plot the heatmap
    heatplot(
        mtx=mtx,
        fig_path=result_path + "_wr_max",
        xlabels=opponent_labels_consistent[1:],
        ylabels=opponent_labels_consistent,
    )

    # Average win rate

    mtx = T.zeros((len(opponent_labels), len(opponent_labels) - 1))
    for name, df in dfs.items():
        training_opponent_label, evaluation_opponent_label = name.split("_to_")
        row = opponent_labels.index(training_opponent_label)
        col = opponent_labels.index(evaluation_opponent_label) - 1

        mtx[row, col] = df["win_rateMean"].mean()

    # Plot the heatmap
    heatplot(
        mtx=mtx,
        fig_path=result_path + "_wr_avg",
        xlabels=opponent_labels_consistent[1:],
        ylabels=opponent_labels_consistent,
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
