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
        # We add a STAND to the N and B attacks to avoid charging the special move
        FootsiesMove.N_ATTACK:      ((False, False, True), (False, False, False)),
        FootsiesMove.B_ATTACK:      ((False, True, True), (False, False, False)),
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
    TEMPORAL_ACTIONS_INT: set[int] = None
    # Moves that signify a player is hit. The opponent is able to cancel into another action in these cases. Note that GUARD_PROXIMITY is not included
    HIT_GUARD_STATES = {FootsiesMove.DAMAGE, FootsiesMove.GUARD_STAND, FootsiesMove.GUARD_CROUCH, FootsiesMove.GUARD_M, FootsiesMove.GUARD_BREAK}
    # Neutral moves on which players can act, since they are instantaneous (I think guard proximity also counts?)
    NEUTRAL_STATES = {FootsiesMove.STAND, FootsiesMove.BACKWARD, FootsiesMove.FORWARD, FootsiesMove.GUARD_PROXIMITY}
    # Temporal moves that can be canceled into other moves
    TEMPORAL_ACTIONS_CANCELABLE = {FootsiesMove.N_ATTACK, FootsiesMove.B_ATTACK}

    ## Conversions between actions

    @staticmethod
    def discrete_to_primitive(discrete: int) -> tuple[bool, bool, bool]:
        """Convert a discrete (integer) action into a primitive (boolean combination) action."""
        return ((discrete & 1) != 0, (discrete & 2) != 0, (discrete & 4) != 0)

    @staticmethod
    def primitive_to_discrete(primitive: tuple[bool, bool, bool]) -> int:
        """Convert a primitive (boolean combination) action into a discrete (integer) action."""
        return (primitive[0] << 0) + (primitive[1] << 1) + (primitive[2] << 2)

    @staticmethod
    def simple_as_move_to_discrete(simple: FootsiesMove) -> Iterable[int]:
        """Covert a simple action (as a `FootsiesMove`) into a sequence of discrete (integer) actions."""
        return ActionMap.SIMPLE_AS_MOVE_TO_DISCRETE_MAP[simple]
    
    @staticmethod
    def simple_as_move_to_primitive(simple: FootsiesMove) -> Iterable[tuple[bool, bool, bool]]:
        """Convert a simple action (as a `Footsiesmove`) into a sequence of primitive (boolean combination) actions."""
        return ActionMap.SIMPLE_AS_MOVE_TO_PRIMITIVE_MAP[simple]

    @staticmethod
    def simple_to_discrete(simple: int) -> Iterable[int]:
        """Convert a simple action (as an integer) into a sequence of discrete (integer) actions."""
        return ActionMap.SIMPLE_TO_DISCRETE_MAP[simple]
    
    @staticmethod
    def simple_to_primitive(simple: int) -> Iterable[tuple[bool, bool, bool]]:
        """Convert a simple action (as an integer) into a sequence of primitive (boolean combination) actions."""
        return ActionMap.SIMPLE_TO_PRIMITIVE_MAP[simple]

    ## Extractions

    @staticmethod
    def move_from_move_index(move_index: int) -> FootsiesMove:
        """Extract the `FootsiesMove` from the move index."""
        return FOOTSIES_MOVE_INDEX_TO_MOVE[move_index]

    # NOTE: the below two methods classify the N_SPECIAL from N/B_ATTACK as being N_SPECIAL rather than N/B_ATTACK
    @staticmethod
    def simple_from_move_index(move_index: int) -> int:
        """Obtain the simple action corresponding to the given move index. \n\n NOTE: `N_SPECIAL` is not correctly classified when canceling into it from `N/B_ATTACK`."""
        return ActionMap.SIMPLE_FROM_MOVE_INDEX_MAP[move_index]

    @staticmethod
    def simple_from_move(move: FootsiesMove) -> int:
        """Obtain the simple action corresponding to the given `FootsiesMove`. \n\n NOTE: `N_SPECIAL` is not correctly classified when canceling into it from `N/B_ATTACK`"""
        return ActionMap.SIMPLE_FROM_MOVE_MAP[move]

    # TODO: the very first frame of temporal actions is being counted as actionable
    @staticmethod
    def simple_from_transition(previous_player_move_index: int, previous_opponent_move_index: int, previous_player_move_progress: float, previous_opponent_move_progress: float, player_move_index: int) -> int:
        """Correctly infer the simple action that was effectively performed in a game transition. If the action was ineffectual, return `None`. This is the method that should be used for obtaining simple actions from gameplay."""
        previous_player_move = FOOTSIES_MOVE_INDEX_TO_MOVE[previous_player_move_index]
        previous_opponent_move = FOOTSIES_MOVE_INDEX_TO_MOVE[previous_opponent_move_index]
        player_move = FOOTSIES_MOVE_INDEX_TO_MOVE[player_move_index]
        was_actionable = ActionMap.is_state_actionable(previous_player_move, previous_opponent_move, previous_player_move_progress, previous_opponent_move_progress)
        
        # The player should have been able to perform an action, otherwise no simple action was effectively performed
        if was_actionable:
            # If the N/B_ATTACK was canceled into N_SPECIAL, we should return the N_ATTACK simple action
            if previous_player_move in ActionMap.TEMPORAL_ACTIONS_CANCELABLE and player_move == FootsiesMove.N_SPECIAL:
                return ActionMap.simple_from_move(FootsiesMove.N_ATTACK) if was_actionable else None
            
            # Otherwise, return the simple action extracted directly from the move
            else:
                return ActionMap.simple_from_move(player_move) if was_actionable else None
        
        return None
    
    @staticmethod
    def simple_as_move(simple: int) -> FootsiesMove:
        """Obtain the `FootsiesMove` that identifies the given simple action."""
        return ActionMap.SIMPLE_ACTIONS[simple]

    ## Attributes

    @staticmethod
    def n_simple() -> int:
        """Number of simple actions."""
        return len(ActionMap.SIMPLE_ACTIONS)

    ## Utility methods

    @staticmethod
    def is_in_hitstop(player_move_state: FootsiesMove, opponent_move_state: FootsiesMove, opponent_move_progress: float) -> bool:
        """Whether the player, at the current move state, is in hitstop."""
        return (
            # The player should be performing an action that takes time and is cancelable
            player_move_state in ActionMap.TEMPORAL_ACTIONS_CANCELABLE
            # Opponent has just been hit
            and opponent_move_progress == 0.0 and opponent_move_state in ActionMap.HIT_GUARD_STATES
            # (
                # The move progress is either not advancing (frozen in time)...
                # np.isclose(previous_move_progress, current_move_progress)
                # ... or that move was canceled into another move. We hardcode the only situation in which this happens (if I'm not wrong)
                # or current_player_move_state == FootsiesMove.N_SPECIAL
            # )
        )

    @staticmethod
    def is_state_actionable(player_move_state: FootsiesMove, opponent_move_state: FootsiesMove, player_move_progress: float, opponent_move_progress: float) -> bool:
        """Whether the player, at the current move state, is able to perform an action."""
        in_hitstop = ActionMap.is_in_hitstop(player_move_state, opponent_move_state, opponent_move_progress)
        return (
            # Is the player in hitstop? (performing an action that takes time and the opponent was just hit)
            in_hitstop
            # Current move is finishing
            or np.isclose(player_move_state.value.duration * player_move_progress + 1, player_move_state.value.duration)
            # Is the player in a neutral state? i.e. states from which the player can always act
            or player_move_state in ActionMap.NEUTRAL_STATES
        )
    
    @staticmethod
    def is_simple_action_commital(simple: int) -> bool:
        """Whether the simple action has a set duration and presents risks"""
        return simple in ActionMap.TEMPORAL_ACTIONS_INT


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
ActionMap.TEMPORAL_ACTIONS_INT = {
    ActionMap.SIMPLE_ACTIONS.index(simple)
    for simple in ActionMap.TEMPORAL_ACTIONS
}