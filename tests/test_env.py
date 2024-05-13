import torch
from footsies_gym.envs.footsies import FootsiesEnv
from agents.utils import FootsiesPhasicMoveProgress
from gymnasium.wrappers.transform_observation import TransformObservation
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.spaces import MultiDiscrete
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.normalization import FootsiesNormalized
from functools import partial
from tests.conftest import wrap_env


def test_phasic_move_progress_wrapper(footsies_env: FootsiesEnv):
    env = wrap_env(footsies_env, [
        FootsiesNormalized,
        FootsiesPhasicMoveProgress,
        FlattenObservation,
        FootsiesActionCombinationsDiscretized,
        partial(TimeLimit, max_episode_steps=300),
        partial(TransformObservation, f=lambda o: torch.from_numpy(o).float().unsqueeze(0)),
    ])

    # Standard size + (remove default move progress + add phasic move progress) * each player
    total_obs_size = 36 + (-1 + 3) * 2

    obs, _ = env.reset()
    assert obs.size(1) == total_obs_size

    terminated, truncated = False, False
    while not (terminated or truncated):
        # NOTE: this is wrong! the action space needs a seed *explicitly* set for reproducibility
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        assert obs.size(1) == total_obs_size


def test_unnormalized_guard_wrapper(footsies_env: FootsiesEnv):
    env = wrap_env(footsies_env, [
        partial(FootsiesNormalized, normalize_guard=False),
        FlattenObservation,
        FootsiesActionCombinationsDiscretized,
        partial(TimeLimit, max_episode_steps=300),
        partial(TransformObservation, f=lambda o: torch.from_numpy(o).float().unsqueeze(0)),
    ])

    assert isinstance(env, TransformObservation)

    e = env
    found = False
    while e != footsies_env:
        if isinstance(e, FootsiesNormalized):
            assert isinstance(e.observation_space.spaces["guard"], MultiDiscrete)
            found = True
            break
        e = e.env
    assert found

    # Standard size + (remove normalized guard + add non-normalized guard) * each player
    total_obs_size = 36 + (-1 + 4) * 2

    obs, _ = env.reset()
    assert obs.size(1) == total_obs_size

    terminated, truncated = False, False
    while not (terminated or truncated):
        obs, _, terminated, truncated, _ = env.step((False, False, False))
        assert obs.size(1) == total_obs_size

