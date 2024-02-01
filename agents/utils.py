from copy import copy
from typing import Callable, Tuple
from gymnasium import Env, ObservationWrapper, ActionWrapper
from stable_baselines3.common.base_class import BaseAlgorithm

# Some wrappers need to be handled in a special manner when extracting a policy for the FOOTSIES environment
from footsies_gym.wrappers.frame_skip import FootsiesFrameSkipped
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.moves import FootsiesMove, footsies_move_index_to_move


# "Action moves" are defined as moves that correspond to intentional actions of the players, i.e. they are a direct result of the player's actions
FOOTSIES_ACTION_MOVES: list[FootsiesMove] = [FootsiesMove.STAND, FootsiesMove.BACKWARD, FootsiesMove.FORWARD, FootsiesMove.DASH_BACKWARD, FootsiesMove.DASH_FORWARD, FootsiesMove.N_ATTACK, FootsiesMove.B_ATTACK, FootsiesMove.N_SPECIAL, FootsiesMove.B_SPECIAL]
# TODO: this DAMAGE -> STAND conversion is valid? Will it not disturb training of STAND? evaluate that
# This mapping maps a FOOTSIES move to an action move that may have caused it. For DAMAGE, we assume the player did nothing
FOOTSIES_ACTION_MOVE_MAP: dict[FootsiesMove, FootsiesMove] = {
    FootsiesMove.STAND: FootsiesMove.STAND,
    FootsiesMove.FORWARD: FootsiesMove.FORWARD,
    FootsiesMove.BACKWARD: FootsiesMove.BACKWARD,
    FootsiesMove.DASH_BACKWARD: FootsiesMove.DASH_BACKWARD,
    FootsiesMove.DASH_FORWARD: FootsiesMove.DASH_FORWARD,
    FootsiesMove.N_ATTACK: FootsiesMove.N_ATTACK,
    FootsiesMove.B_ATTACK: FootsiesMove.B_ATTACK,
    FootsiesMove.N_SPECIAL: FootsiesMove.N_SPECIAL,
    FootsiesMove.B_SPECIAL: FootsiesMove.B_SPECIAL,
    FootsiesMove.DAMAGE: FootsiesMove.STAND,
    FootsiesMove.GUARD_M: FootsiesMove.BACKWARD,
    FootsiesMove.GUARD_STAND: FootsiesMove.BACKWARD,
    FootsiesMove.GUARD_CROUCH: FootsiesMove.BACKWARD,
    FootsiesMove.GUARD_BREAK: FootsiesMove.BACKWARD,
    FootsiesMove.GUARD_PROXIMITY: FootsiesMove.BACKWARD,
    FootsiesMove.DEAD: FootsiesMove.STAND,
    FootsiesMove.WIN: FootsiesMove.STAND,
}
FOOTSIES_ACTION_MOVE_INDEX_MAP: dict[FootsiesMove, int] = {
    move: FOOTSIES_ACTION_MOVES.index(FOOTSIES_ACTION_MOVE_MAP[move]) for move in FootsiesMove
}
FOOTSIES_ACTION_MOVE_INDICES_MAP: dict[int, int] = {
    footsies_move_index_to_move.index(move): FOOTSIES_ACTION_MOVES.index(FOOTSIES_ACTION_MOVE_MAP[move]) for move in FootsiesMove
}

assert set(FOOTSIES_ACTION_MOVES) == set(FOOTSIES_ACTION_MOVE_MAP.values()) and len(FOOTSIES_ACTION_MOVE_MAP) == len(FootsiesMove)


def wrap_policy(
    env: Env, internal_policy: Callable
) -> Callable[[dict], Tuple[bool, bool, bool]]:
    observation_wrappers = []
    footsies_observation_wrappers = []  # these need to be applied before frameskipping
    action_wrappers = []

    frameskip_wrapper = None

    current_env = env
    while current_env != current_env.unwrapped:
        if isinstance(current_env, ObservationWrapper):
            if isinstance(current_env, FootsiesNormalized):
                footsies_observation_wrappers.append(current_env)
            else:
                observation_wrappers.append(current_env)

        elif isinstance(current_env, ActionWrapper):
            action_wrappers.append(current_env)

        elif isinstance(current_env, FootsiesFrameSkipped):
            frameskip_wrapper = current_env

        current_env = current_env.env

    def policy(obs: dict) -> Tuple[bool, bool, bool]:
        for footsies_observation_wrapper in reversed(footsies_observation_wrappers):
            obs = footsies_observation_wrapper.observation(obs)

        # TODO: not the best solution, the condition is always evaluated even though it has always the same value
        # NOTE: it's assumed that the frameskip wrapper is wrapped by any other observation/action wrappers, except those for FOOTSIES
        if frameskip_wrapper is not None:
            if frameskip_wrapper._is_obs_skippable(obs):
                return (False, False, False)

            obs = frameskip_wrapper._frame_skip_obs(obs)

        for observation_wrapper in reversed(observation_wrappers):
            obs = observation_wrapper.observation(obs)

        action = internal_policy(obs)

        for action_wrapper in action_wrappers:
            action = action_wrapper.action(action)

        return action

    return policy


def snapshot_sb3_policy(agent: BaseAlgorithm, deterministic: bool = False):
    policy = copy(agent.policy)
    policy.load_state_dict(agent.policy.state_dict())

    def wrapper(obs):
        return policy.predict(obs, deterministic=deterministic)[0].item()

    return wrapper
