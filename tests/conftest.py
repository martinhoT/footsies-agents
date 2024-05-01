import pytest
import torch
import gymnasium as gym
from footsies_gym.envs.footsies import FootsiesEnv
from data import FootsiesDataset, FootsiesTorchDataset
from torch.utils.data import DataLoader
from agents.action import ActionMap
from gymnasium.wrappers.transform_observation import TransformObservation
from gymnasium.wrappers.flatten_observation import FlattenObservation
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.normalization import FootsiesNormalized
from functools import partial
from contextlib import contextmanager


@pytest.fixture(scope="function", autouse=True)
def init_torch():
    torch.manual_seed(0)


@pytest.fixture(scope="package")
def il_dataset():
    dataset = FootsiesDataset.load("footsies-dataset")
    dataset = FootsiesTorchDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    def dataset_iterator():
        for obs, next_obs, reward, p1_action, p2_action, terminated in dataloader:
            obs = obs.float()
            p1_action, p2_action = ActionMap.simples_from_transition_torch(obs, next_obs)

            yield obs, next_obs, reward, p1_action, p2_action, terminated

    return dataset_iterator


@pytest.fixture(scope="package")
def footsies_env_root():
    env = FootsiesEnv(
        game_path="../Footsies-Gym/Build/FOOTSIES.x86_64",
        game_port=15000,
        opponent_port=15001,
        remote_control_port=15002,
        render_mode=None,
        sync_mode="synced_non_blocking",
        fast_forward=False,
        dense_reward=False,
        opponent=lambda o, i: (False, False, False),
    )
    
    # Set in-game bot as default opponent
    env.set_opponent(None)

    yield env

    env.close()


@pytest.fixture(scope="function")
def footsies_env(footsies_env_root: FootsiesEnv):
    env = footsies_env_root

    yield env

    env.set_opponent(None)
    env.reset()


def wrap_env(footsies_env: FootsiesEnv, wrappers: list[gym.Wrapper]) -> gym.Env:
    env = footsies_env
    for wrapper in wrappers:
        env = wrapper(env)
    return env


def wrap_env_standard(footsies_env: FootsiesEnv) -> gym.Env:
    return wrap_env(footsies_env, [
        FootsiesNormalized,
        FlattenObservation,
        FootsiesActionCombinationsDiscretized,
        partial(TransformObservation, f=lambda o: torch.from_numpy(o).float().unsqueeze(0)),
    ])
