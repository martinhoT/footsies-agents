import multiprocessing as mp
from os import path
from scripts.evaluation.utils import create_env
from models import to_
from itertools import combinations
from copy import deepcopy
from opponents.curriculum import Idle, Backer, NSpammer, BSpammer, NSpecialSpammer, BSpecialSpammer, WhiffPunisher, CurriculumOpponent
from typing import cast, Callable
from agents.the_one.agent import TheOneAgent
from dataclasses import dataclass
from opponents.base import Opponent
from tqdm import trange
from agents.action import ActionMap

OPPONENT_MAP: dict[str, Callable[[], CurriculumOpponent | None]] = {
    "idle": Idle,
    "backer": Backer,
    "n_spammer": NSpammer,
    "b_spammer": BSpammer,
    "n_special_spammer": NSpecialSpammer,
    "b_special_spammer": BSpecialSpammer,
    "whiff_punisher": WhiffPunisher,
    "bot": lambda: None,
}

@dataclass
class AgentRun:
    name:                   str
    agent:                  TheOneAgent
    training_opponent:      Opponent | None
    evaluation_opponent:    Opponent | None


def train_agent_against_opponent(agent: TheOneAgent, opponent: Opponent | None, id_: int, label: str = "", timesteps: int = int(1e6), initial_seed: int | None = 0):
    port_start = 11000 + 1000 * id_
    port_stop = 11000 + (1000) * (id_ + 1)
    env, footsies_env = create_env(port_start=port_start, port_stop=port_stop)

    o = opponent.act if opponent is not None else None
    footsies_env.set_opponent(o)

    seed = initial_seed
    terminated, truncated = True, True
    for step in trange(timesteps, desc=label, unit="step", dynamic_ncols=True, colour="#FF9E64"):
        if (terminated or truncated):
            obs, info = env.reset(seed=seed)
            terminated, truncated = False, False
            seed = None

        action = agent.act(obs, info)
        next_obs, reward, terminated, truncated, next_info = env.step(action)

        # Skip hitstop freeze
        in_hitstop = ActionMap.is_in_hitstop_ori(next_info, True) or ActionMap.is_in_hitstop_ori(next_info, False)
        while in_hitstop and obs.isclose(next_obs).all():
            action = agent.act(obs, info)
            next_obs, r, terminated, truncated, next_info = env.step(action)
            reward += r

        agent.update(obs, next_obs, reward, terminated, truncated, info, next_info)

        obs, info = next_obs, next_info

def main(
    seeds: int = 10,
    timesteps: int = int(1e6),
    processes: int = 4,
    y: bool = False,
):
    dummy_env, _  = create_env()

    agent, _ = to_(
        observation_space_size=dummy_env.observation_space.shape[0],
        action_space_size=dummy_env.action_space.n,
    )

    opponent_labels = ["blank", "idle", "backer", "n_spammer", "b_spammer", "n_special_spammer", "b_special_spammer", "whiff_punisher", "bot"]

    agent_runs: list[AgentRun] = [
        AgentRun(
            name=f"{training_opponent}_to_{evaluation_opponent}",
            agent=deepcopy(agent),
            training_opponent=OPPONENT_MAP[training_opponent](),
            evaluation_opponent=OPPONENT_MAP[evaluation_opponent](),
        )
        for training_opponent, evaluation_opponent in combinations(opponent_labels, 2)
        if training_opponent is not "blank"
    ]

    with mp.Pool(processes=processes) as pool:
        pool.starmap(train_agent_against_opponent, [run.agent, run.map])

    agents = [(run.name, run.agent, run.evaluation_opponent) for run in agent_runs]

    get_data(
        data="win_rate",
        agents=agents,
        title="Win rate over the last 100 episodes against the in-game bot",
        fig_path=path.splitext(__file__)[0],
        seeds=seeds,
        timesteps=1000000,
        exp_factor=0.9,
        xlabel="Time step",
        ylabel="Win rate",
        run_name_mapping={
            "discount_1_0":             "$\\gamma = 1.0$",
            "discount_0_999":           "$\\gamma = 0.999$",
            "discount_0_99":            "$\\gamma = 0.99$",
            "discount_0_9":             "$\\gamma = 0.9$",
            "discount_1_0_correct":     "$\\gamma = 1.0$ (correct)",
            "discount_0_999_correct":   "$\\gamma = 0.999$ (correct)",
            "discount_0_99_correct":    "$\\gamma = 0.99$ (correct)",
            "discount_0_9_correct":     "$\\gamma = 0.9$ (correct)",
        }
    )

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
