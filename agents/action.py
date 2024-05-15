import torch as T
from typing import Sequence
from footsies_gym.moves import FootsiesMove, FOOTSIES_MOVE_INDEX_TO_MOVE


class ActionMap:
    """
    Class containing useful methods and data structures regarding FOOTSIES's actions.
    
    There are 3 types of actions:
    - Primitive, the most basic FOOTSIES action representation, as a triple of bools (<left>, <right>, <attack>)
    - Discrete, representing primitive actions as integers
    - Simple, representing actionable `FootsiesMove` moves as integers

    This class contains methods for converting between these 3 types of actions.
    """

    ## Data structures
    ## Some data structures are initialized after class definition

    # "Simple moves" are defined as moves that correspond to intentional actions of the players, i.e. they are a direct result of the player's actions
    SIMPLE_ACTIONS: list[FootsiesMove] = [FootsiesMove.STAND, FootsiesMove.FORWARD, FootsiesMove.BACKWARD, FootsiesMove.DASH_FORWARD, FootsiesMove.DASH_BACKWARD, FootsiesMove.N_ATTACK, FootsiesMove.B_ATTACK, FootsiesMove.N_SPECIAL, FootsiesMove.B_SPECIAL]

    # Assuming player on the left side
    SIMPLE_AS_MOVE_TO_PRIMITIVE_MAP: dict[FootsiesMove, Sequence[tuple[bool, bool, bool]]] = {
        # STAND clears the possibility of performing a dash by inputting backward and forward
        FootsiesMove.STAND:         ((True, True, False),),
        FootsiesMove.FORWARD:       ((False, True, False),),
        FootsiesMove.BACKWARD:      ((True, False, False),),
        FootsiesMove.DASH_FORWARD:  ((False, False, False), (False, True, False), (False, False, False), (False, True, False)),
        FootsiesMove.DASH_BACKWARD: ((False, False, False), (True, False, False), (False, False, False), (True, False, False)),
        # We add a STAND to the N and B attacks to avoid charging the special move
        FootsiesMove.N_ATTACK:      ((False, False, True), (False, False, False)),
        FootsiesMove.B_ATTACK:      ((False, True, True), (False, False, False)),
        FootsiesMove.N_SPECIAL:     ((False, False, True),) * 59 + ((False, False, False),),
        FootsiesMove.B_SPECIAL:     ((False, False, True),) * 59 + ((False, True, False),),
    }
    SIMPLE_AS_MOVE_TO_DISCRETE_MAP: dict[FootsiesMove, Sequence[int]] = {}
    SIMPLE_TO_PRIMITIVE_MAP: dict[int, Sequence[tuple[bool, bool, bool]]] = {}
    SIMPLE_TO_DISCRETE_MAP: dict[int, Sequence[int]] = {}
    
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
    SIMPLE_FROM_MOVE_MAP: dict[FootsiesMove, int] = {}
    SIMPLE_FROM_MOVE_INDEX_MAP: dict[int, int] = {}

    # Actions that have a set duration, and cannot be performed instantly. These actions are candidates for frame skipping
    TEMPORAL_ACTIONS = set(SIMPLE_ACTIONS) - {FootsiesMove.STAND, FootsiesMove.FORWARD, FootsiesMove.BACKWARD}
    TEMPORAL_ACTIONS_INT: set[int] = set()
    # Moves that signify a player is hit. The opponent is able to cancel into another action in these cases. Note that GUARD_PROXIMITY is not included
    HIT_GUARD_STATES = {FootsiesMove.DAMAGE, FootsiesMove.GUARD_STAND, FootsiesMove.GUARD_CROUCH, FootsiesMove.GUARD_M, FootsiesMove.GUARD_BREAK}
    HIT_GUARD_STATES_INT: set[int] = set()
    # Neutral moves on which players can act, since they are instantaneous (I think guard proximity also counts?)
    NEUTRAL_STATES = {FootsiesMove.STAND, FootsiesMove.BACKWARD, FootsiesMove.FORWARD, FootsiesMove.GUARD_PROXIMITY}
    # Temporal moves that can be canceled into other moves
    # NOTE: the method `is_in_hitstop_torch` hardcodes this set and assumes to contain these moves. If it is to be updated, update that method as well
    TEMPORAL_STATES_CANCELABLE = {FootsiesMove.N_ATTACK, FootsiesMove.B_ATTACK}
    TEMPORAL_STATES_CANCELABLE_INT: set[int] = set()
    # Simple actions that are performable in hitstop
    PERFORMABLE_SIMPLES_IN_HITSTOP = {FootsiesMove.STAND, FootsiesMove.N_ATTACK}
    PERFORMABLE_SIMPLES_IN_HITSTOP_INT: set[int] = set()
    # Simple actions that are technically special moves (i.e. require a sequence of primitive actions to perform)
    SIMPLE_SPECIAL_MOVES = {FootsiesMove.DASH_FORWARD, FootsiesMove.DASH_BACKWARD, FootsiesMove.N_SPECIAL, FootsiesMove.B_SPECIAL}
    SIMPLE_SPECIAL_MOVES_INT: set[int] = set()

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
    def simple_as_move_to_discrete(simple: FootsiesMove) -> Sequence[int]:
        """Covert a simple action (as a `FootsiesMove`) into a sequence of discrete (integer) actions."""
        return ActionMap.SIMPLE_AS_MOVE_TO_DISCRETE_MAP[simple]
    
    @staticmethod
    def simple_as_move_to_primitive(simple: FootsiesMove) -> Sequence[tuple[bool, bool, bool]]:
        """Convert a simple action (as a `Footsiesmove`) into a sequence of primitive (boolean combination) actions."""
        return ActionMap.SIMPLE_AS_MOVE_TO_PRIMITIVE_MAP[simple]

    @staticmethod
    def simple_to_discrete(simple: int) -> Sequence[int]:
        """Convert a simple action (as an integer) into a sequence of discrete (integer) actions."""
        return ActionMap.SIMPLE_TO_DISCRETE_MAP[simple]
    
    @staticmethod
    def simple_to_primitive(simple: int) -> Sequence[tuple[bool, bool, bool]]:
        """Convert a simple action (as an integer) into a sequence of primitive (boolean combination) actions."""
        return ActionMap.SIMPLE_TO_PRIMITIVE_MAP[simple]

    @staticmethod
    def invert_simple(simple: int) -> int:
        """Invert the given simple action, done in one player's perspective, into the other player's. It's recommended to use the other inversion methods rather than this one."""
        simple_move = ActionMap.simple_as_move(simple)
        if simple_move == FootsiesMove.FORWARD:
            simple = ActionMap.simple_from_move(FootsiesMove.BACKWARD)
        elif simple_move == FootsiesMove.BACKWARD:
            simple = ActionMap.simple_from_move(FootsiesMove.FORWARD)
        elif simple_move == FootsiesMove.DASH_FORWARD:
            simple = ActionMap.simple_from_move(FootsiesMove.DASH_BACKWARD)
        elif simple_move == FootsiesMove.DASH_BACKWARD:
            simple = ActionMap.simple_from_move(FootsiesMove.DASH_FORWARD)
        return simple

    @staticmethod
    def invert_discrete(discrete: int) -> int:
        """Invert the given discrete action, done in one player's perspective, into the other player's."""
        # Invert the left and right actions
        return ((discrete & 1) << 1) + ((discrete & 2) >> 1) + (discrete & 4)

    @staticmethod
    def invert_primitive(primitive: tuple[bool, bool, bool]) -> tuple[bool, bool, bool]:
        """Invert the given primitive action, done in one player's perspective, into the other player's."""
        # Invert the left and right actions
        return (primitive[1], primitive[0], primitive[2])

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

    # NOTE: the very first frame of temporal actions is the action that originated that temporal action
    @staticmethod
    def simple_from_transition(previous_player_move_index: int, previous_opponent_move_index: int, previous_player_move_frame: int, previous_opponent_move_frame: int, player_move_index: int) -> int | None:
        """Correctly infer the simple action that was effectively performed in a game transition. If the action was ineffectual, return `None`. This is the method that should be used for obtaining simple actions from gameplay."""
        previous_player_move = FOOTSIES_MOVE_INDEX_TO_MOVE[previous_player_move_index]
        previous_opponent_move = FOOTSIES_MOVE_INDEX_TO_MOVE[previous_opponent_move_index]
        player_move = FOOTSIES_MOVE_INDEX_TO_MOVE[player_move_index]
        was_actionable = ActionMap.is_state_actionable(previous_player_move, previous_opponent_move, previous_player_move_frame, previous_opponent_move_frame)
        
        # The player should have been able to perform an action, otherwise no simple action was effectively performed
        if was_actionable:
            # If the N/B_ATTACK was canceled into N_SPECIAL, we should return the N_ATTACK simple action
            if previous_player_move in ActionMap.TEMPORAL_STATES_CANCELABLE:
                if player_move == FootsiesMove.N_SPECIAL:
                    return ActionMap.simple_from_move(FootsiesMove.N_ATTACK)
                # However, be careful! If the player is in hitstop, then we can't assume they are performing N_ATTACK here.
                # Rather, we assume they are doing nothing unless they perform a valid move for this cancellation
                else:
                    return ActionMap.simple_from_move(FootsiesMove.STAND)
            
            # Otherwise, return the simple action extracted directly from the move
            else:
                return ActionMap.simple_from_move(player_move)
        
        return None
    
    @staticmethod
    def simples_from_transition_ori(obs: dict, next_obs: dict) -> tuple[int | None, int | None]:
        """Correctly infer the simple actions from player 1 and 2 that were performed in the given game transition as original observations. If an action was ineffectual, return `None`. This is a convenience method that should be used for obtaining simple actions from gameplay."""
        p1_move_frame = obs["move_frame"][0]
        p2_move_frame = obs["move_frame"][1]
        p1_move_index = obs["move"][0]
        p2_move_index = obs["move"][1]
        p1_next_move_index = next_obs["move"][0]
        p2_next_move_index = next_obs["move"][1]
        
        obs_agent_action = ActionMap.simple_from_transition(
            previous_player_move_index=p1_move_index,
            previous_opponent_move_index=p2_move_index,
            previous_player_move_frame=p1_move_frame,
            previous_opponent_move_frame=p2_move_frame,
            player_move_index=p1_next_move_index,
        )
        obs_opponent_action = ActionMap.simple_from_transition(
            previous_player_move_index=p2_move_index,
            previous_opponent_move_index=p1_move_index,
            previous_player_move_frame=p2_move_frame,
            previous_opponent_move_frame=p1_move_frame,
            player_move_index=p2_next_move_index,
        )

        return obs_agent_action, obs_opponent_action
    
    @staticmethod
    def simples_from_transition_torch(obs: T.Tensor, next_obs: T.Tensor) -> tuple[int | None, int | None]:
        """
        Correctly infer the simple actions from player 1 and 2 that were performed in the given game transition as PyTorch tensors.
        If an action was ineffectual, return `None`.
        This is a convenience method that should be used for obtaining simple actions from gameplay.
        No batch support.
        """
        if obs.size(0) > 1 or next_obs.size(0) > 1:
            raise ValueError("batched observations are not supported for this method")

        p1_move_index = int(T.argmax(obs[0, 2:17]).item())
        p2_move_index = int(T.argmax(obs[0, 17:32]).item())
        p1_move = ActionMap.move_from_move_index(p1_move_index)
        p2_move = ActionMap.move_from_move_index(p2_move_index)
        p1_next_move_index = int(T.argmax(next_obs[0, 2:17]).item())
        p2_next_move_index = int(T.argmax(next_obs[0, 17:32]).item())
        p1_move_progress = obs[0, 32].item()
        p2_move_progress = obs[0, 33].item()
        p1_move_frame = round(p1_move_progress * p1_move.value.duration)
        p2_move_frame = round(p2_move_progress * p2_move.value.duration)
        obs_agent_action = ActionMap.simple_from_transition(
            previous_player_move_index=p1_move_index,
            previous_opponent_move_index=p2_move_index,
            previous_player_move_frame=p1_move_frame,
            previous_opponent_move_frame=p2_move_frame,
            player_move_index=p1_next_move_index,
        )
        obs_opponent_action = ActionMap.simple_from_transition(
            previous_player_move_index=p2_move_index,
            previous_opponent_move_index=p1_move_index,
            previous_player_move_frame=p2_move_frame,
            previous_opponent_move_frame=p1_move_frame,
            player_move_index=p2_next_move_index,
        )

        return obs_agent_action, obs_opponent_action

    @staticmethod
    def simple_as_move(simple: int) -> FootsiesMove:
        """Obtain the `FootsiesMove` that identifies the given simple action."""
        return ActionMap.SIMPLE_ACTIONS[simple]

    ## Attributes

    @staticmethod
    def n_simple() -> int:
        """Number of simple actions."""
        return len(ActionMap.SIMPLE_ACTIONS)

    @staticmethod
    def is_simple_action_special_move(simple: int) -> bool:
        """Whether the given simple action is a special move."""
        return simple in ActionMap.SIMPLE_SPECIAL_MOVES_INT

    ## Utility methods

    @staticmethod
    def is_in_hitstop(player_move_state: FootsiesMove, opponent_move_state: FootsiesMove, opponent_move_frame: int) -> bool:
        """Whether the player, at the current move state, is in hitstop."""
        return (
            # The player should be performing an action that takes time and is cancelable
            player_move_state in ActionMap.TEMPORAL_STATES_CANCELABLE
            # TODO: Does this work with hitting the opponent again while they are guarding?
            # Opponent has just been hit
            and opponent_move_frame == 0 and opponent_move_state in ActionMap.HIT_GUARD_STATES
        )

    @staticmethod
    def is_in_hitstop_ori(obs: dict, p1: bool = True) -> bool:
        """Whether the player, at the given original observation, is in hitstop."""
        idx = 0 if p1 else 1
        opponent_move_frame = obs["move_frame"][1 - idx]
        player_move_index = obs["move"][idx]
        opponent_move_index = obs["move"][1 - idx]
        player_move_state = FOOTSIES_MOVE_INDEX_TO_MOVE[player_move_index]
        opponent_move_state = FOOTSIES_MOVE_INDEX_TO_MOVE[opponent_move_index]
        return ActionMap.is_in_hitstop(player_move_state, opponent_move_state, opponent_move_frame)

    _TEMPORAL_STATES_CANCELABLE_TORCH: T.Tensor = T.tensor(0)
    _HIT_GUARD_STATES_TORCH: T.Tensor = T.tensor(0)

    @staticmethod
    def is_in_hitstop_torch(obs: T.Tensor, p1: bool = True) -> T.Tensor:
        """Whether the player, at the given PyTorch tensor observation, is in hitstop. Supports a batch of observations."""
        idx = 0 if p1 else 1
        
        player_move_index = obs[:, 2 + idx * 15:17 + idx * 15].argmax(dim=-1)
        opponent_move_index = obs[:, 17 - idx * 15:32 - idx * 15].argmax(dim=-1)

        agent_in_temporal_action = T.isin(player_move_index, ActionMap._TEMPORAL_STATES_CANCELABLE_TORCH, assume_unique=True)

        # We won't expect floating point errors here, since zero should be exactly zero
        opponent_at_start = obs[:, 33 - idx] == 0.0

        opponent_in_hitguard = T.isin(opponent_move_index, ActionMap._HIT_GUARD_STATES_TORCH, assume_unique=True)

        return agent_in_temporal_action & opponent_at_start & opponent_in_hitguard

    @staticmethod
    def is_state_actionable(player_move_state: FootsiesMove, opponent_move_state: FootsiesMove, player_move_frame: int, opponent_move_frame: int) -> bool:
        """Whether the player, at the current move state, is able to perform an action."""
        in_hitstop = ActionMap.is_in_hitstop(player_move_state, opponent_move_state, opponent_move_frame)
        return (
            # Is the player in hitstop? (performing an action that takes time and the opponent was just hit)
            in_hitstop
            # Current move is finishing
            or player_move_frame + 1 == player_move_state.value.duration
            # Is the player in a neutral state? i.e. states from which the player can always act
            or player_move_state in ActionMap.NEUTRAL_STATES
        )
    
    @staticmethod
    def is_state_actionable_ori(obs: dict, p1: bool = True) -> bool:
        """Whether the player, at the current move state, is able to perform an action, from the given original observation."""
        idx = 0 if p1 else 1
        player_move_frame = obs["move_frame"][idx]
        opponent_move_frame = obs["move_frame"][1 - idx]
        player_move_index = obs["move"][idx]
        opponent_move_index = obs["move"][1 - idx]
        player_move_state = ActionMap.move_from_move_index(player_move_index)
        opponent_move_state = ActionMap.move_from_move_index(opponent_move_index)
        return ActionMap.is_state_actionable(player_move_state, opponent_move_state, player_move_frame, opponent_move_frame)

    @staticmethod
    def is_state_actionable_torch(obs: T.Tensor, p1: bool = True) -> bool:
        """Whether the player, at the current move state, is able to perform an action, from the given PyTorch observation. No batch support."""
        if obs.size(0) > 1:
            raise ValueError("batched observations are not supported for this method")

        idx = 0 if p1 else 1
        player_move_progress = obs[0, 32 + idx]
        opponent_move_progress = obs[0, 33 - idx]
        player_move_index = int(obs[0, 2 + idx * 15:17 + idx * 15].argmax(dim=-1).item())
        opponent_move_index = int(obs[0, 17 - idx * 15:32 - idx * 15].argmax(dim=-1).item())
        player_move_state = ActionMap.move_from_move_index(player_move_index)
        opponent_move_state = ActionMap.move_from_move_index(opponent_move_index)
        
        player_move_frame = int((player_move_progress * player_move_state.value.duration).round().item())
        opponent_move_frame = int((opponent_move_progress * opponent_move_state.value.duration).round().item())

        return ActionMap.is_state_actionable(player_move_state, opponent_move_state, player_move_frame, opponent_move_frame)

    @staticmethod
    def is_at_neutral_actionable_torch(obs: T.Tensor, p1: bool = True) -> bool:
        """The same as `is_state_actionable` but without counting hitstop"""
        if obs.size(0) > 1:
            raise ValueError("batched observations are not supported for this method")

        idx = 0 if p1 else 1
        player_move_progress = obs[0, 32 + idx]
        player_move_index = int(obs[0, 2 + idx * 15:17 + idx * 15].argmax(dim=-1).item())
        player_move_state = ActionMap.move_from_move_index(player_move_index)
        
        player_move_frame = int((player_move_progress * player_move_state.value.duration).round().item())
        
        return (
            # Current move is finishing
            player_move_frame + 1 == player_move_state.value.duration
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
ActionMap.HIT_GUARD_STATES_INT = {
    FOOTSIES_MOVE_INDEX_TO_MOVE.index(move)
    for move in ActionMap.HIT_GUARD_STATES
}
ActionMap.TEMPORAL_STATES_CANCELABLE_INT = {
    FOOTSIES_MOVE_INDEX_TO_MOVE.index(move) for move in ActionMap.TEMPORAL_STATES_CANCELABLE
}
ActionMap.PERFORMABLE_SIMPLES_IN_HITSTOP_INT = {
    ActionMap.SIMPLE_ACTIONS.index(simple) for simple in ActionMap.PERFORMABLE_SIMPLES_IN_HITSTOP
}
ActionMap.SIMPLE_SPECIAL_MOVES_INT = {
    ActionMap.SIMPLE_ACTIONS.index(simple) for simple in ActionMap.SIMPLE_SPECIAL_MOVES
}
ActionMap._TEMPORAL_STATES_CANCELABLE_TORCH = T.tensor(list(ActionMap.TEMPORAL_STATES_CANCELABLE_INT)).long()
ActionMap._HIT_GUARD_STATES_TORCH = T.tensor(list(ActionMap.HIT_GUARD_STATES_INT)).long()

assert all(v is not None for v in vars(ActionMap).values())
