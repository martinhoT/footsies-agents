import logging
from collections import deque
from typing import Any, Literal, SupportsFloat, cast
from gymnasium import Env, ObservationWrapper, Wrapper, spaces
from agents.action import ActionMap
from footsies_gym.moves import FootsiesMove
from footsies_gym.envs.footsies import FootsiesEnv
from torch.utils.tensorboard import SummaryWriter # type: ignore
from footsies_gym.wrappers.normalization import FootsiesNormalized
from opponents.base import OpponentManager
from opponents.curriculum import CurriculumOpponent


class AppendSimpleHistoryWrapper(Wrapper):
    """Observation wrapper for appending the action history of one of the players. Must be put after `FootsiesNormalized`."""
    def __init__(self, env, p1: bool, n: int, distinct: bool):
        super().__init__(env)

        e = env
        has_normalized = False
        while e != e.unwrapped:
            if isinstance(e, FootsiesNormalized):
                has_normalized = True
                break
            e = e.env
        
        if not has_normalized:
            raise ValueError("AppendSimpleHistoryWrapper must be put after FootsiesNormalized")

        self.observation_space: spaces.Dict = env.observation_space
        self.observation_space.spaces[f"p{1 if p1 else 2}_history"] = spaces.MultiDiscrete([ActionMap.n_simple()] * n)

        self.p1 = p1
        self.distinct = distinct
        # Fill history with no-ops
        self.history = deque([0] * n, maxlen=n)
        self.history_len = n

        self.prev_obs = None
    
    def observation(self, obs: dict) -> dict:
        obs = obs.copy()
        
        # Determine which action the player performed into `obs`
        if self.prev_obs is not None:
            action = ActionMap.simple_from_transition(
                previous_player_move_index=self.prev_obs["move"][0 if self.p1 else 1],
                previous_opponent_move_index=self.prev_obs["move"][1 if self.p1 else 0],
                previous_player_move_frame=self.prev_obs["move_frame"][0 if self.p1 else 1],
                previous_opponent_move_frame=self.prev_obs["move_frame"][1 if self.p1 else 0],
                player_move_index=obs["move"][0 if self.p1 else 1],
            )

            if action is not None and (not self.distinct or (self.history[-1] != action)):
                self.history.append(action)

        self.prev_obs = obs

        # Append the history of actions so far to the observation
        obs[f"p{1 if self.p1 else 2}_history"] = list(self.history)

        return obs

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict]:
        self.history.clear()
        self.history.extend([0] * self.history_len)
        self.prev_obs = None

        obs, info = self.env.reset(seed=seed, options=options)
        obs = self.observation(obs)

        return obs, info

    def step(self, action) -> tuple[Any, SupportsFloat, bool, bool, dict]:
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        next_obs = self.observation(next_obs)

        return next_obs, reward, terminated, truncated, info
    

class FootsiesEncourageAdvance(Wrapper):
    """Wrapper for providing a reward for advancing towards the opponent based on the distance between players."""
    
    def __init__(self, env,
        distance_cap: float = 4.0,
        advance_reward: float = 0.02,
        log_dir: str | None = None,
    ):
        """
        Wrapper for providing a reward for advancing towards the opponent based on the distance between players.
        
        Parameters
        ----------
        - `distance_cap`: the maximum distance beyond which 0 reward will be given. The default corresponds to round start position
        - `advance_reward`: the maximum reward for advancing towards the opponent, which will decrease towards 0 the greater the distance is between the players
        - `log_dir`: the directory where to save the Tensorboard logs regarding how much extra reward was given on each episode
        """
        super().__init__(env)

        self.footsies_env: FootsiesEnv = env.unwrapped

        self.distance_cap = distance_cap
        self.advance_reward = advance_reward

        # For logging
        self._summary_writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None
        self._episode_extra_reward = 0.0
        self._episodes = 0

    def step(self, action: tuple[bool, bool, bool]) -> tuple[dict, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)

        obs_original = self.footsies_env.most_recent_observation

        distance_between_players = min(obs_original["position"][1] - obs_original["position"][0], self.distance_cap)

        # Reward agent for advancing forward, and close to the opponent
        p1_move = ActionMap.move_from_move_index(obs_original["move"][0])
        if p1_move == FootsiesMove.FORWARD:
            extra = self.advance_reward * (self.distance_cap - distance_between_players) / self.distance_cap
            self._episode_extra_reward += extra
            reward += extra

        # Logging
        if terminated or truncated:
            if self._summary_writer is not None:
                self._summary_writer.add_scalar(
                    "Performance/Reward from advancing",
                    self._episode_extra_reward,
                    self._episodes,
                )

            self._episode_extra_reward = 0.0
            self._episodes += 1

        return obs, reward, terminated, truncated, info


class FootsiesPhasicMoveProgress(ObservationWrapper):
    """Wrapper for bucketizing the move progress according to the move's phase (startup, active, recovery). Must be put after `FootsiesNormalized`, since it works with unflattened observations and normalized move progress."""

    def __init__(self, env: Env):
        super().__init__(env)

        self.observation_space = cast(spaces.Dict, env.observation_space)
        self.observation_space.spaces["move_frame"] = spaces.MultiDiscrete([3, 3]) # type: ignore

    def observation(self, obs: dict) -> dict:
        obs = obs.copy()

        p1_move = ActionMap.move_from_move_index(obs["move"][0])
        p2_move = ActionMap.move_from_move_index(obs["move"][1])

        p1_move_frame = round(obs["move_frame"][0] * p1_move.value.duration)
        p2_move_frame = round(obs["move_frame"][1] * p2_move.value.duration)
        p1_move_phase = 2 if p1_move.in_recovery(p1_move_frame) else 1 if p1_move.in_active(p1_move_frame) else 0
        p2_move_phase = 2 if p2_move.in_recovery(p2_move_frame) else 1 if p2_move.in_active(p2_move_frame) else 0

        obs["move_frame"] = (p1_move_phase, p2_move_phase)

        return obs


class FootsiesSimpleActionExecutor:
    """Class for executing simple actions (integers) as primitive actions on FOOTSIES."""

    LOGGER = logging.getLogger("main.wrapper.simple-action-executor")

    def __init__(self, allow_special_moves: bool):
        self._allow_special_moves = allow_special_moves
        self._simple_action = None
        self._simple_action_queue = deque([])

    def act(self, simple_action: int):
        """
        Request the simple action to be executed and return the primitive action that should be performed now on FOOTSIES.
        If another simple action is still being performed, then this one will be ignored.
        """
        if not self._allow_special_moves:
            # Convert the detected special move input to a simple action.
            if simple_action == 8 or simple_action == 7:
                self.LOGGER.warning("We detected the agent performing a special move, even though they can't perform special moves! Will convert to the respective attack action.")
                simple_action -= 2

        if self._simple_action_queue:
            primitive_action = self._simple_action_queue.popleft()

        else:
            self._simple_action = simple_action
            self._simple_action_queue.extend(ActionMap.simple_to_primitive(simple_action))
            primitive_action = self._simple_action_queue.popleft()

        return primitive_action

    def reset(self):
        """Reset the executor state, which should be done everytime the environment terminates/truncates."""
        self._simple_action = None
        self._simple_action_queue.clear()


class FootsiesSimpleActionExtractor:
    """Class for extracting information regarding simple actions from FOOTSIES."""

    def __init__(self,
            assumed_agent_action_on_nonactionable: Literal["last", "none", "stand"],
            assumed_opponent_action_on_nonactionable: Literal["last", "none", "stand"],
        ):
        self._assumed_agent_action_on_nonactionable: Literal["last", "none", "stand"] = assumed_agent_action_on_nonactionable
        self._assumed_opponent_action_on_nonactionable: Literal["last", "none", "stand"] = assumed_opponent_action_on_nonactionable

        self._last_valid_p1_action = 0
        self._last_valid_p2_action = 0
        self._current_info = {}
    
    def update(self, info: dict) -> dict:
        info = info.copy()
        inferred_p1_action, inferred_p2_action = ActionMap.simples_from_transition_ori(self._current_info, info)

        info["p1_simple"] = self.effective_action(inferred_p1_action, self._assumed_agent_action_on_nonactionable, self._last_valid_p1_action)
        info["p2_simple"] = self.effective_action(inferred_p2_action, self._assumed_opponent_action_on_nonactionable, self._last_valid_p2_action)
        info["p1_was_actionable"] = inferred_p1_action is not None
        info["p2_was_actionable"] = inferred_p2_action is not None
        info["p1_is_actionable"] = ActionMap.is_state_actionable_ori(info, p1=True)
        info["p2_is_actionable"] = ActionMap.is_state_actionable_ori(info, p1=False)
        self._current_info = info
        if inferred_p1_action is not None:
            self._last_valid_p1_action = inferred_p1_action
        if inferred_p2_action is not None:
            self._last_valid_p2_action = inferred_p2_action
        
        return info

    def reset(self, info: dict) -> dict:
        """Reset the extractor state, which should be done everytime the environment terminates/truncates. Also, return the updated info dictionary."""
        info = info.copy()

        info["p1_simple"] = 0
        info["p2_simple"] = 0
        info["p1_was_actionable"] = False
        info["p2_was_actionable"] = False
        info["p1_is_actionable"] = True
        info["p2_is_actionable"] = True
        self._current_info = info
        self._last_valid_p1_action = 0
        self._last_valid_p2_action = 0

        return info

    def effective_action(self, inferred_action: int | None, nonactionable_resolution_method: Literal["last", "none", "stand"], last_valid_action: int) -> int | None:
        """Get the effective action that was performed from observation. If obtained in an nonactionable state, then resolve it according to the arguments."""
        if inferred_action is None:
            if nonactionable_resolution_method == "last":
                return last_valid_action
            elif nonactionable_resolution_method == "stand":
                return 0
            elif nonactionable_resolution_method == "none":
                return None
        
        return inferred_action

    @property
    def current_info(self) -> dict:
        """The current info (the one that was created in the last `reset` or `update` call)."""
        return self._current_info


class FootsiesSimpleActions(Wrapper):
    """
    Wrapper for using simple actions. Appends to the info dictionaries the simple action that each player performed, and correctly executes simple actions from the agent on `step`.

    The agent's and opponent's simplified actions are treated differently.
    For the opponent, we don't care how they performed the action at the low level, we only care about its effect on the state (i.e. the move they are performing).
    
    The agent works a little differently however.
    Since for executing the simple actions ourselves we do need to perform them at the low level, when the agent requests simple action X there is some
    "overhead" involved, which is the sequence of primitive actions required to execute the simple action.
    This has implications on the Markov property that environment states should ideally have.
    As such, the move information of the agent is changed to incorporate this.

    For example: the agent wants to perform `DASH_FORWARD`, so the sequence is `[(0, 1, 0), (0, 0, 0), (0, 1, 0)]`.
    As a result, the environment state will have `DASH_FORWARD` as the move the moment the first primitive sequence starts, and it will be considered to have 2 more frames of duration.

    TLDR: we assume the opponent does simple actions instantly; we consider the agent needs some more timesteps to execute one
    """

    def __init__(self, env,
        agent_allow_special_moves: bool = True,
        assumed_agent_action_on_nonactionable: Literal["last", "none", "stand"] = "last",
        assumed_opponent_action_on_nonactionable: Literal["last", "none", "stand"] = "last",
    ):
        """
        Wrapper for using simple actions.

        Parameters
        ----------
        - `agent_allow_special_moves`: whether to allow the agent the ability to perform special moves
        - `assumed_{agent,opponent}_action_on_nonactionable`: when frameskipping (i.e. the agent's or opponent's action is `None`), we can do one of three things:
            - Consider the agent/opponent to be doing an invalid action, corresponding to `"none"`. This usually means models consider "any" action
            - Consider the agent/opponent to still be doing the last valid action before frameskipping (update on a single action), corresponding to `"last"`
            - Consider the agent/opponent to do the `STAND` no-op action (0), corresponding to `"stand"`
        
        If using tabular models, it's recommended to use "none", since it's technically more correct.
        However, if using a function approximator such as a neural network, it's likely that considering "any" action (i.e. broadcasting) will leak into the state in which the original action
        was performed (since states are similar), which will muddy the function's output.
        In such cases, it's recommended to use either `"last"` or `"stand"`.
        """
        super().__init__(env)

        if assumed_agent_action_on_nonactionable not in ("last", "stand", "none"):
            raise ValueError("'assumed_agent_action_on_nonactionable' must be either 'last', 'stand' or 'none'")
        if assumed_opponent_action_on_nonactionable not in ("last", "stand", "none"):
            raise ValueError("'assumed_opponent_action_on_nonactionable' must be either 'last', 'stand' or 'none'")

        self._executor = FootsiesSimpleActionExecutor(
            allow_special_moves=agent_allow_special_moves
        )
        self._extractor = FootsiesSimpleActionExtractor(
            assumed_agent_action_on_nonactionable=assumed_agent_action_on_nonactionable,
            assumed_opponent_action_on_nonactionable=assumed_opponent_action_on_nonactionable
        )
        
        self._agent_allow_special_moves = agent_allow_special_moves
        self._assumed_agent_action_on_nonactionable: Literal["last", "none", "stand"] = assumed_agent_action_on_nonactionable
        self._assumed_opponent_action_on_nonactionable: Literal["last", "none", "stand"] = assumed_opponent_action_on_nonactionable
        
        n_simple = ActionMap.n_simple()

        self.action_space = spaces.Discrete(n_simple - (2 if not agent_allow_special_moves else 0))
        self._opponent_action_space = spaces.Discrete(n_simple)
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        
        self._executor.reset()
        info = self._extractor.reset(info)
        info["agent_simple"] = 0
        info["agent_simple_completed"] = True

        return obs, info

    def step(self, simple_action: int) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        primitive_action = self._executor.act(simple_action)

        obs, reward, terminated, truncated, info = self.env.step(primitive_action)

        info = self._extractor.update(info)
        info["agent_simple"] = self._executor._simple_action
        info["agent_simple_completed"] = len(self._executor._simple_action_queue) == 0

        return obs, reward, terminated, truncated, info

    @property
    def current_agent_simple_action(self) -> int:
        return self._agent_simple_action

    @property
    def opponent_action_space(self) -> spaces.Discrete:
        """The opponent's action space."""
        return self._opponent_action_space

    @property
    def assumed_agent_action_on_nonactionable(self) -> Literal["last", "stand", "none"]:
        """Which agent action to assume when it can't act."""
        return self._assumed_agent_action_on_nonactionable
    
    @property
    def assumed_opponent_action_on_nonactionable(self) -> Literal["last", "stand", "none"]:
        """Which opponent action to assume when it can't act."""
        return self._assumed_opponent_action_on_nonactionable

    @property
    def agent_allow_special_moves(self) -> bool:
        """Whether to allow the agent the ability to perform special moves."""
        return self._agent_allow_special_moves


class OpponentManagerWrapper(Wrapper):
    """
    Wrapper that includes an opponent manager into the environment interaction.
    Mainly used for applying opponent managers, such as the curriculum, to SB3 algorithms.
    May only be applied to FOOTSIES (`FootsiesEnv`).
    """

    def __init__(self, env, opponent_manager: OpponentManager):
        super().__init__(env)

        self.env = env
        self.footsies_env = cast(FootsiesEnv, env.unwrapped)
        self._opponent_manager = opponent_manager
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict]:
        if self._opponent_manager.current_opponent is not None:
            self._opponent_manager.current_opponent.reset()

        # This makes sure the environment and the curriculum manager are in sync
        opponent = self._opponent_manager.current_opponent.act if self._opponent_manager.current_opponent is not None else None
        self.footsies_env.set_opponent(opponent)

        obs, info = self.env.reset(seed=seed, options=options)

        return obs, info

    def step(self, action: int) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Whether to notify the agent of the opponent's next action distribution
        if isinstance(self._opponent_manager.current_opponent, CurriculumOpponent):
            info["next_opponent_policy"] = self._opponent_manager.current_opponent.peek(info)

        if terminated or truncated:
            # Determine the final game result, to provide to the self-play manager
            result = 0.5
            if terminated and (reward != 0.0):
                result = 1 if float(reward) > 0.0 else 0
            elif truncated and (info["guard"][0] != info["guard"][1]):
                result = 1 if info["guard"][0] > info["guard"][1] else 0

            # Set a new opponent from the opponent pool
            should_change = self._opponent_manager.update_at_episode(result)

            if should_change:
                opponent = self._opponent_manager.current_opponent.act if self._opponent_manager.current_opponent is not None else None
                self.footsies_env.set_opponent(opponent)

        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()
        self._opponent_manager.close()

    @property
    def opponent_manager(self) -> OpponentManager:
        return self._opponent_manager
