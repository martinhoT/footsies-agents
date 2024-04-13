import torch
from typing import Callable, TypeVar
from agents.base import FootsiesAgentBase
from gymnasium import Env
from gymnasium.wrappers.transform_observation import TransformObservation
from gymnasium.wrappers.flatten_observation import FlattenObservation
from agents.wrappers import FootsiesSimpleActions
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers import FootsiesNormalized
from agents.the_one.agent import TheOneAgent
from agents.action import ActionMap
from tqdm import trange


def dummy_opponent(o: dict, i: dict) -> tuple[bool, bool, bool]:
    return (False, False, False)

def obs_to_torch(o) -> torch.Tensor:
    return torch.from_numpy(o).float().unsqueeze(0)

def create_env(
    port_start: int = 5000,
    port_stop: int | None = None,
) -> tuple[Env, FootsiesEnv]:
    
    footsies_env = FootsiesEnv(
        game_path="../Footsies-Gym/Build/FOOTSIES.x86_64",
        render_mode=None,
        sync_mode="synced_non_blocking",
        fast_forward=True,
        dense_reward=False,
        opponent=dummy_opponent,
        **FootsiesEnv.find_ports(start=port_start, stop=port_stop),
    )

    env = TransformObservation(
        FootsiesSimpleActions(
            FlattenObservation(
                FootsiesNormalized(
                    footsies_env,
                )
            )
        ),
        obs_to_torch,
    )

    return env, footsies_env


class Observer:
    def update(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict, agent: FootsiesAgentBase):
        pass


T = TypeVar("T", bound=Observer)

def test(agent: TheOneAgent, label: str, id_: int, observer_type: type[T], opponent: Callable[[dict, dict], tuple[bool, bool, bool]] | None = None, episodes: int = 1000) -> T:
    port_start = 11000 + 100 * id_
    port_stop = 11000 + (100) * (id_ + 1)
    env, footsies_env = create_env(port_start=port_start, port_stop=port_stop)

    footsies_env.set_opponent(opponent)

    observer = observer_type()

    for _ in trange(episodes, desc=label, unit="ep", position=id_, dynamic_ncols=True, colour="#80e4ed"):
        obs, info = env.reset()
        terminated, truncated = False, False

        while not (terminated or truncated):
            action = agent.act(obs, info)
            next_obs, reward, terminated, truncated, next_info = env.step(action)

            # Skip hitstop freeze
            in_hitstop = ActionMap.is_in_hitstop_ori(next_info, True) or ActionMap.is_in_hitstop_ori(next_info, False)
            while in_hitstop and obs.isclose(next_obs).all():
                action = agent.act(obs, info)
                next_obs, r, terminated, truncated, next_info = env.step(action)
                reward += r

            agent.update(obs, next_obs, reward, terminated, truncated, info, next_info)
            observer.update(obs, next_obs, reward, terminated, truncated, info, next_info, agent)

            obs, info = next_obs, next_info
    
    return observer