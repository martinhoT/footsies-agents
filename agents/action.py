import numpy as np
from typing import Iterable
from footsies_gym.moves import FootsiesMove, FOOTSIES_MOVE_INDEX_TO_MOVE


class ActionMap:
    """
    Class containing useful methods and data structures regarding FOOTSIES's actions.
    
    There are 3 types of actions:
    - Primitive, the most basic FOOTSIES action representation, as a triple of bools
    - Discrete, representing primitive actions as integers
    - Simple, representing actionable `FootsiesMove` moves as integers

    This class contains methods for converting between these 3 types of actions.
    """

    ## Data structures
    ## Some data structures are initialized after class definition

    # "Simple moves" are defined as moves that correspond to intentional actions of the players, i.e. they are a direct result of the player's actions
    SIMPLE_ACTIONS: list[FootsiesMove] = [FootsiesMove.STAND, FootsiesMove.FORWARD, FootsiesMove.BACKWARD, FootsiesMove.DASH_FORWARD, FootsiesMove.DASH_BACKWARD, FootsiesMove.N_ATTACK, FootsiesMove.B_ATTACK, FootsiesMove.N_SPECIAL, FootsiesMove.B_SPECIAL]

    # Assuming player on the left side
    SIMPLE_AS_MOVE_TO_PRIMITIVE_MAP: dict[FootsiesMove, Iterable[tuple[bool, bool, bool]]] = {
        FootsiesMove.STAND:         ((False, False, False),),
        FootsiesMove.FORWARD:       ((False, True, False),),
        FootsiesMove.BACKWARD:      ((True, False, False),),
        FootsiesMove.DASH_FORWARD:  ((False, True, False), (False, False, False), (False, True, False)),
        FootsiesMove.DASH_BACKWARD: ((True, False, False), (False, False, False), (True, False, False)),
        FootsiesMove.N_ATTACK:      ((False, False, True),),
        FootsiesMove.B_ATTACK:      ((False, True, True),),
        FootsiesMove.N_SPECIAL:     tuple(((False, False, True) for _ in range(59))) + ((False, False, False),),
        FootsiesMove.B_SPECIAL:     tuple(((False, False, True) for _ in range(59))) + ((False, True, False),),
    }
    SIMPLE_AS_MOVE_TO_DISCRETE_MAP: dict[FootsiesMove, Iterable[int]] = None
    SIMPLE_TO_PRIMITIVE_MAP: dict[int, Iterable[tuple[bool, bool, bool]]] = None
    SIMPLE_TO_DISCRETE_MAP: dict[int, Iterable[int]] = None

    # TODO: is this DAMAGE -> STAND conversion valid? Will it not disturb training of STAND? evaluate that
    # This mapping maps a FOOTSIES move to an action move that may have caused it. For DAMAGE, we assume the player did nothing
    SIMPLE_AS_MOVE_FROM_MOVE_MAP: dict[FootsiesMove, FootsiesMove] = {
        FootsiesMove.STAND:             FootsiesMove.STAND,
        FootsiesMove.FORWARD:           FootsiesMove.FORWARD,
        FootsiesMove.BACKWARD:          FootsiesMove.BACKWARD,
        FootsiesMove.DASH_FORWARD:      FootsiesMove.DASH_FORWARD,
        FootsiesMove.DASH_BACKWARD:     FootsiesMove.DASH_BACKWARD,
        FootsiesMove.N_ATTACK:          FootsiesMove.N_ATTACK,
        FootsiesMove.B_ATTACK:          FootsiesMove.B_ATTACK,
        FootsiesMove.N_SPECIAL:         FootsiesMove.N_SPECIAL,
        FootsiesMove.B_SPECIAL:         FootsiesMove.B_SPECIAL,
        FootsiesMove.DAMAGE:            FootsiesMove.STAND,
        FootsiesMove.GUARD_M:           FootsiesMove.BACKWARD,
        FootsiesMove.GUARD_STAND:       FootsiesMove.BACKWARD,
        FootsiesMove.GUARD_CROUCH:      FootsiesMove.BACKWARD,
        FootsiesMove.GUARD_BREAK:       FootsiesMove.BACKWARD,
        FootsiesMove.GUARD_PROXIMITY:   FootsiesMove.BACKWARD,
        FootsiesMove.DEAD:              FootsiesMove.STAND,
        FootsiesMove.WIN:               FootsiesMove.STAND,
    }
    SIMPLE_FROM_MOVE_MAP: dict[FootsiesMove, int] = None
    SIMPLE_FROM_MOVE_INDEX_MAP: dict[int, int] = None

    # Actions that have a set duration, and cannot be performed instantly. These actions are candidates for frame skipping
    TEMPORAL_ACTIONS = set(SIMPLE_ACTIONS) - {FootsiesMove.STAND, FootsiesMove.FORWARD, FootsiesMove.BACKWARD}
    # Moves that signify a player is hit. The opponent is able to cancel into another action in these cases. Note that GUARD_PROXIMITY is not included
    HIT_GUARD_STATES = {FootsiesMove.DAMAGE, FootsiesMove.GUARD_STAND, FootsiesMove.GUARD_CROUCH, FootsiesMove.GUARD_M, FootsiesMove.GUARD_BREAK}
    # Neutral moves on which players can act, since they are instantaneous (I think guard proximity also counts?)
    NEUTRAL_STATES = {FootsiesMove.STAND, FootsiesMove.BACKWARD, FootsiesMove.FORWARD, FootsiesMove.GUARD_PROXIMITY}

    ## Conversions between actions

    @staticmethod
    def discrete_to_primitive(discrete: int) -> tuple[bool, bool, bool]:
        return ((discrete & 1) != 0, (discrete & 2) != 0, (discrete & 4) != 0)

    @staticmethod
    def primitive_to_discrete(primitive: tuple[bool, bool, bool]) -> int:
        return (primitive[0] << 0) + (primitive[1] << 1) + (primitive[2] << 2)

    @staticmethod
    def simple_as_move_to_discrete(simple: FootsiesMove) -> Iterable[int]:
        return ActionMap.SIMPLE_AS_MOVE_TO_DISCRETE_MAP[simple]
    
    @staticmethod
    def simple_as_move_to_primitive(simple: FootsiesMove) -> Iterable[tuple[bool, bool, bool]]:
        return ActionMap.SIMPLE_AS_MOVE_TO_PRIMITIVE_MAP[simple]

    @staticmethod
    def simple_to_discrete(simple: int) -> Iterable[int]:
        return ActionMap.SIMPLE_TO_DISCRETE_MAP[simple]
    
    @staticmethod
    def simple_to_primitive(simple: int) -> Iterable[tuple[bool, bool, bool]]:
        return ActionMap.SIMPLE_TO_PRIMITIVE_MAP[simple]

    ## Extractions

    @staticmethod
    def move_from_move_index(move_index: int) -> FootsiesMove:
        return FOOTSIES_MOVE_INDEX_TO_MOVE[move_index]

    @staticmethod
    def simple_from_move_index(move_index: int) -> int:
        return ActionMap.SIMPLE_FROM_MOVE_INDEX_MAP[move_index]

    @staticmethod
    def simple_from_move(move: FootsiesMove) -> int:
        return ActionMap.SIMPLE_FROM_MOVE_MAP[move]

    @staticmethod
    def simple_as_move(simple: int) -> FootsiesMove:
        return ActionMap.SIMPLE_ACTIONS[simple]

    ## Attributes

    @staticmethod
    def n_simple() -> int:
        return len(ActionMap.SIMPLE_ACTIONS)

    ## Utility methods

    # TODO: the very first hit frame is not being counted as hitstop
    @staticmethod
    def is_in_hitstop_late(previous_player_move_state: FootsiesMove, previous_move_progress: float, current_move_progress: float) -> bool:
        """Whether the player, at the previous move state, is in hitstop. This evaluation is done late, as it can't be done on the current state"""
        return previous_player_move_state in ActionMap.TEMPORAL_ACTIONS and np.isclose(previous_move_progress, current_move_progress) and current_move_progress > 0.0

    @staticmethod
    def is_state_actionable_late(previous_player_move_state: FootsiesMove, previous_move_progress: float, current_move_progress: float) -> bool:
        """Whether the player, at the previous move state, is able to perform an action. This evaluation is done late, as it can't be done on the current state"""
        in_hitstop = ActionMap.is_in_hitstop_late(previous_player_move_state, previous_move_progress, current_move_progress)
        return (
            # Is the player in hitstop? (performing an action that takes time and the opponent was just hit)
            in_hitstop
            # Previous move has just finished
            or current_move_progress < previous_move_progress
            # Is the player in a neutral state?
            or previous_player_move_state in ActionMap.NEUTRAL_STATES
        )


assert set(ActionMap.SIMPLE_ACTIONS) == set(ActionMap.SIMPLE_AS_MOVE_FROM_MOVE_MAP.values()) and len(ActionMap.SIMPLE_AS_MOVE_FROM_MOVE_MAP) == len(FootsiesMove)
assert set(ActionMap.SIMPLE_ACTIONS) == set(ActionMap.SIMPLE_AS_MOVE_TO_PRIMITIVE_MAP.keys())


ActionMap.SIMPLE_AS_MOVE_TO_DISCRETE_MAP = {
    simple: tuple(ActionMap.primitive_to_discrete(primitive) for primitive in ActionMap.SIMPLE_AS_MOVE_TO_PRIMITIVE_MAP[simple])
    for simple in ActionMap.SIMPLE_ACTIONS
}
ActionMap.SIMPLE_TO_PRIMITIVE_MAP = {
    ActionMap.SIMPLE_ACTIONS.index(simple): ActionMap.SIMPLE_AS_MOVE_TO_PRIMITIVE_MAP[simple]
    for simple in ActionMap.SIMPLE_ACTIONS
}
ActionMap.SIMPLE_TO_DISCRETE_MAP = {
    ActionMap.SIMPLE_ACTIONS.index(simple): ActionMap.SIMPLE_AS_MOVE_TO_DISCRETE_MAP[simple]
    for simple in ActionMap.SIMPLE_ACTIONS
}
ActionMap.SIMPLE_FROM_MOVE_MAP = {
    move: ActionMap.SIMPLE_ACTIONS.index(ActionMap.SIMPLE_AS_MOVE_FROM_MOVE_MAP[move]) for move in FootsiesMove
}
ActionMap.SIMPLE_FROM_MOVE_INDEX_MAP = {
    FOOTSIES_MOVE_INDEX_TO_MOVE.index(move): ActionMap.SIMPLE_ACTIONS.index(ActionMap.SIMPLE_AS_MOVE_FROM_MOVE_MAP[move]) for move in FootsiesMove
}