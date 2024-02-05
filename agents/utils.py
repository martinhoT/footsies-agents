import numpy as np
from copy import copy
from typing import Callable, Tuple
from gymnasium import Env, ObservationWrapper, ActionWrapper
from stable_baselines3.common.base_class import BaseAlgorithm

# Some wrappers need to be handled in a special manner when extracting a policy for the FOOTSIES environment
from footsies_gym.wrappers.frame_skip import FootsiesFrameSkipped
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.moves import FootsiesMove, footsies_move_index_to_move


# "Action moves" are defined as moves that correspond to intentional actions of the players, i.e. they are a direct result of the player's actions
FOOTSIES_ACTION_MOVES: list[FootsiesMove] = [FootsiesMove.STAND, FootsiesMove.FORWARD, FootsiesMove.BACKWARD, FootsiesMove.DASH_FORWARD, FootsiesMove.DASH_BACKWARD, FootsiesMove.N_ATTACK, FootsiesMove.B_ATTACK, FootsiesMove.N_SPECIAL, FootsiesMove.B_SPECIAL]
# TODO: this DAMAGE -> STAND conversion is valid? Will it not disturb training of STAND? evaluate that
# This mapping maps a FOOTSIES move to an action move that may have caused it. For DAMAGE, we assume the player did nothing
FOOTSIES_ACTION_MOVE_MAP: dict[FootsiesMove, FootsiesMove] = {
    FootsiesMove.STAND: FootsiesMove.STAND,
    FootsiesMove.FORWARD: FootsiesMove.FORWARD,
    FootsiesMove.BACKWARD: FootsiesMove.BACKWARD,
    FootsiesMove.DASH_FORWARD: FootsiesMove.DASH_FORWARD,
    FootsiesMove.DASH_BACKWARD: FootsiesMove.DASH_BACKWARD,
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

# Actions that have a set duration, and cannot be performed instantly. These actions are candidates for frame skipping
TEMPORAL_ACTIONS = set(FOOTSIES_ACTION_MOVES) - {FootsiesMove.STAND, FootsiesMove.FORWARD, FootsiesMove.BACKWARD}
# Moves that signify a player is hit. The opponent is able to cancel into another action in these cases. Note that GUARD_PROXIMITY is not included
HIT_GUARD_STATES = {FootsiesMove.DAMAGE, FootsiesMove.GUARD_STAND, FootsiesMove.GUARD_CROUCH, FootsiesMove.GUARD_M, FootsiesMove.GUARD_BREAK}
# Neutral moves on which players can act, since they are instantaneous (I think guard proximity also counts?)
NEUTRAL_STATES = {FootsiesMove.STAND, FootsiesMove.BACKWARD, FootsiesMove.FORWARD, FootsiesMove.GUARD_PROXIMITY}


# TODO: the very first hit frame is not being counted as hitstop
def is_in_hitstop_late(previous_player_move_state: FootsiesMove, previous_move_progress: float, current_move_progress: float) -> bool:
    """Whether the player, at the previous move state, is in hitstop. This evaluation is done late, as it can't be done on the current state"""
    return previous_player_move_state in TEMPORAL_ACTIONS and np.isclose(previous_move_progress, current_move_progress) and current_move_progress > 0.0


# TODO: case where a move is done one after another not being counted
def is_state_actionable_late(previous_player_move_state: FootsiesMove, previous_move_progress: float, current_move_progress: float) -> bool:
    """Whether the player, at the previous move state, is able to perform an action. This evaluation is done late, as it can't be done on the current state"""
    in_hitstop = is_in_hitstop_late(previous_player_move_state, previous_move_progress, current_move_progress)
    return (
        # Is the player in hitstop? (performing an action that takes time and the opponent was just hit)
        in_hitstop
        # Previous move has just finished
        or current_move_progress < previous_move_progress
        # Is the player in a neutral state?
        or previous_player_move_state in NEUTRAL_STATES
    )


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
