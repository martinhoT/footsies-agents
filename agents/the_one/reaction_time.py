import torch
import numpy as np
import logging
from typing import Literal, TypeVar
from collections import deque
from itertools import islice
from math import log
from agents.action import ActionMap
from agents.a2c.a2c import ActorNetwork
from agents.game_model.game_model import GameModel
from agents.mimic.mimic import PlayerModel
from functools import cached_property


LOGGER = logging.getLogger("main.the_one.reaction_time")


class ReactionTimeEmulator:
    def __init__(
        self,
        actor: ActorNetwork,
        opponent: PlayerModel,
        multiplier: float = 1,
        additive: float = 0,
        history_size: int = 30,
    ):
        """
        Emulator of human choice reaction time following Hick's law.

        Parameters
        ----------
        - `multiplier`: the entropy multiplier in the reaction time formula
        - `additive`: the additive constant in the reaction time formula
        - `history_size`: the size of the observation history
        """
        self._actor = actor
        self._opponent = opponent
        self._multiplier = multiplier
        self._additive = additive
        self._history_size = history_size
        self._predictor = None

        self._states = deque([], maxlen=history_size)
        self._infos = deque([], maxlen=history_size)
        self._prev_reaction_time = None
        self._constant = False

        self._prev_obs_delayed = None
        self._prev_obs = None
        self._opp_hs = None

    def confine_to_range(self, minimum: int, maximum: int, agent_n_actions: int):
        """
        Define the multiplier and additive parameters to confine the reaction time to a defined range.
        Make sure to apply this operation after other parameters of the emulator, such as the inaction probability, are set.
        """
        if maximum >= self._history_size:
            raise ValueError(f"the maximum value cannot be equal to or greater than the defined history size ({maximum} >= {self._history_size})")
        
        maximum_entropy = self.maximum_decision_entropy(agent_n_actions)
        b = (maximum - minimum) / (maximum_entropy - self._inaction_entropy)
        c = minimum - b * self._inaction_entropy

        self._multiplier = b
        self._additive = c

    def maximum_decision_entropy(self, agent_n_actions: int) -> float:
        """Calculate the maximum possible decision entropy."""
        return log(agent_n_actions)

    def reaction_time(self, decision_entropy: float, previous_reaction_time: int | None = None) -> int:
        """
        Calculate reaction time in time steps given a decision distribution.
        
        Parameters
        ----------
        - `decision_entropy`: the entropy of the probability distribution of the decision
        - `previous_reaction_time`: the reaction time at the previous observation, in order to bound the returned reaction time into a reasonable range.
        The returned reaction time will not be bounded if this value is `None`
        """
        if previous_reaction_time is None:
            previous_reaction_time = self._history_size - 1

        computed_reaction_time = int(np.ceil(self._multiplier * decision_entropy + self._additive))
        # We need to be reasonable, reaction time should not be larger than the previous one (we can'torch.Tensor start seeing things in the past)
        return min(previous_reaction_time + 1, computed_reaction_time)
    
    def register(self, state: torch.Tensor, info: dict):
        """Register a state and agent action into the state and player 1 action histories. Should be performed at every environment step, before perceiving an observation."""
        self._states.append(state)
        self._infos.append(info)
        if self._predictor is not None:
            self._predictor.append_info(info["p1_simple"], info["p2_is_actionable"])
        
        # Make the react() method calculate a new observation again.
        del self.react

    def perceive(self, reaction_time: int, previous_reaction_time: int | None = None) -> tuple[torch.Tensor, list[torch.Tensor], dict, list[dict]]:
        """Get an observation according to the provided reaction time, and all the observations that were skipped according to the previous reaction time. For a reaction time of 0, return the observation at the current instant."""
        if previous_reaction_time is None:
            previous_reaction_time = self._history_size - 1

        observation = self._states[-1 - reaction_time]
        info = self._infos[-1 - reaction_time]
        skipped_observations = list(islice(self._states, (self._states.maxlen - 1) - previous_reaction_time, (self._states.maxlen - 1) - reaction_time))
        skipped_infos = list(islice(self._infos, (self._infos.maxlen - 1) - previous_reaction_time, (self._infos.maxlen - 1) - reaction_time))

        return observation, skipped_observations, info, skipped_infos

    # Having this "cache" stuff allows the "_prev" variables to not be updated everytime react is called, and only after new registrations.
    @cached_property
    def react(self) -> tuple[torch.Tensor, int, torch.Tensor]:
        """
        Perform a reaction, receiving a delayed observation (or a corrected one if a multi-step predictor is used) according to the current reaction time.
        
        This method is cached, and the cache is invalidated when a new state-info pair is registered.

        NOTE: the decision entropy is calculated according to the last reaction time computed through this method, which is initially the maximum possible (perceives the oldest observation).

        Returns
        -------
        - `observation`: the observation that was perceived, `reaction_time` time steps late
        - `reaction_time`: the computed reaction time
        - `opponent_hidden_state`: the hidden state of the opponent model at `observation`, useful for further inference with the opponent model
        """
        # Calculate the decision distribution.
        opp_probs, hs = self._opponent.probabilities(self._prev_obs, self._opp_hs)
        self._opp_hs = hs
        d = self._actor.decision_distribution(self._prev_obs, opp_probs, detached=True)

        # Calculate the reaction time from the decision distribution.
        if self._constant:
            reaction_time = self._additive
        else:
            reaction_time = self.reaction_time(d.entropy().item(), self._prev_reaction_time)

        # Query a delayed observation with the computed reaction time.
        obs, skipped_obs, info, skipped_info = self.perceive(reaction_time, self._prev_reaction_time)

        prev_obs_delayed = self._prev_obs_delayed
        self._prev_obs_delayed = obs

        # If set up, use the delayed observation to predict the current state.
        if self._predictor is not None:
            skipped_state_info_pairs = list(zip(skipped_obs, skipped_info))
            obs, opp_hs = self._predictor.predict(prev_obs_delayed, obs, info, reaction_time, skipped_state_info_pairs)

        self._prev_reaction_time = reaction_time
        self._prev_obs = obs

        return obs, reaction_time, opp_hs

    def reset(self, state: torch.Tensor):
        """Reset the internal state of the reaction time emulator, which should be done at the start of every episode."""
        self._states.clear()
        self._states.extend([state] * self._history_size)
        self._prev_reaction_time = None
        self._prev_obs = state
        self._prev_obs_delayed = state
        del self.react

        if self._predictor is not None:
            self._predictor.reset()

    @property
    def constant(self) -> bool:
        """Whether reaction time is a constant and thus not dependent on the decision distribution's entropy."""
        return self._constant

    @constant.setter
    def constant(self, value: bool):
        self._constant = value

    @property
    def previous_reaction_time(self) -> int:
        """The most recently perceived reaction time (last call to `register_and_perceive`)."""
        return self._prev_reaction_time

    @property
    def history_size(self) -> int:
        """The (maximum) size of the observation history."""
        return self._states.maxlen
    
    @property
    def predictor(self) -> "MultiStepPredictor" | None:
        """The multi-step predictor that may optionally be appended to this emulator for correction of delayed observations into observations at the current time step. If `None`, no correction will be performed."""
        return self._predictor

    @predictor.setter
    def predictor(self, value: "MultiStepPredictor" | None):
        self._predictor = value
    

class MultiStepPredictor:
    def __init__(self,
        reaction_time_emulator: ReactionTimeEmulator,
        game_model: GameModel,
        assumed_opponent_action_on_nonactionable: Literal["last", "none", "stand"] = "last",
        method: Literal["multi-step", "skip-step"] = "skip-step",
    ):
        self._actor = reaction_time_emulator._actor
        self._opponent = reaction_time_emulator._opponent
        self._game_model = game_model
        self._assumed_opponent_action_on_nonactionable = assumed_opponent_action_on_nonactionable
        self._method = method

        # The hidden state of the opponent model (only matters if recurrent).
        # This version of the hidden state is the one calculated after advancing the opponent model up to predicted observation.
        self._current_opp_hs = None
        # The actions that the agent has previously performed.
        # This is a buffer of the same size of the reaction time observation buffer to aid in multi-step prediction of the current state.
        # The buffer doesn'torch.Tensor need to include the action that was performed at the current state, so we subtract 1.
        past_size = reaction_time_emulator.history_size - 1
        self._past_agent_actions = deque([0] * past_size, maxlen=past_size)
        self._past_p2_actionables = deque([True] * past_size, maxlen=past_size)
        # For multi-step prediction, we should make assumptions on the opponent's actions during nonactionable periods
        # in the same way that the environment does, in order to match what the game model expects.
        self._assumed_opponent_action_on_nonactionable = None

    def predict(self, prev_obs: torch.Tensor, obs: torch.Tensor, info: dict, n: int, skipped_state_info_pairs: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Predict the observation `n` timesteps into the future from `obs`.
        
        Returns
        -------
        - `obs`: the predicted current state (i.e. input `obs`), using the delayed state provided by the reaction time emulator
        - `opponent_model_hidden_state`: the hidden state of the opponent model at `obs`, for further prediction
        """
        # Calculate the opponent model hidden state correctly.
        for skipped_state, skipped_info in skipped_state_info_pairs:
            if skipped_info["p2_is_actionable"]:
                self._current_opp_hs = self._opponent.network.compute_hidden_state(skipped_state, self._current_opp_hs)
                if self._opponent.should_reset_context(prev_obs, skipped_state, False):
                    self._current_opp_hs = None
                prev_obs = skipped_state

        if self._opponent.should_reset_context(skipped_state, obs, False):
            self._current_opp_hs = None

        # Correct perceived observation.
        if self._method == "skip-step":
            obs, opponent_model_hidden_state = self.skip_step_prediction(obs, n, self._current_opp_hs, info["p2_simple"], info["p2_is_actionable"])
        elif self._method == "multi-step":
            obs, opponent_model_hidden_state = self.multi_step_prediction(obs, n, self._current_opp_hs, info["p2_simple"])
        else:
            raise RuntimeError(f"the value of 'method' ('{self._method}') is misconfigured! Should be either 'skip-step' or 'multi-step'")
        
        return obs, opponent_model_hidden_state

    def skip_step_prediction(self, obs: torch.Tensor, n: int, opp_hs: torch.Tensor, last_valid_opp_action: torch.Tensor | int, p2_actionable: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the observation `n` time steps into the future, using the agent's past executed actions, opponent model and game model.
        
        Uses a game model that already predicts `n` time steps into the future.
        """
        p1_action = self._past_agent_actions[-n]
        if p2_actionable:
            p2_action, opp_hs = self._opponent.network.probabilities(obs, opp_hs)
        else:
            p2_action = self._resolve_opponent_action(last_valid_opp_action)

        game_model = self._select_apt_game_model(n)
        obs = game_model.predict(obs, p1_action, p2_action)

        return obs

    def multi_step_prediction(self, obs: torch.Tensor, n: int, opp_hs: torch.Tensor, last_valid_opp_action: torch.Tensor | int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the observation `n` time steps into the future, using the agent's past executed actions, opponent model and game model.
        
        Uses a game model that predicts one time step into the future iteratively.
        """
        o_prev = None
        o = obs
        p2_action = last_valid_opp_action

        for t in reversed(range(n)):
            p2_actionable = ActionMap.is_state_actionable_torch(o, p1=False)

            if p2_actionable:
                # If the opponent model hidden state should be reset, according to the reset strategy employed opponent model, do so.
                if o_prev is not None and self._opponent.should_reset_context(o_prev, o, False):
                    opp_hs = None

                p2_action, opp_hs = self._opponent.network.probabilities(o, opp_hs)
            
            else:
                p2_action = self._resolve_opponent_action(p2_action)

            p1_action = self._past_agent_actions[-1 - t]
            
            o_prev = o
            o = self._game_model.predict(o, p1_action, p2_action).detach()
        
        return o, opp_hs

    def _resolve_opponent_action(self, last_valid_action: int) -> int | None:
        if self._assumed_opponent_action_on_nonactionable == "last":
            return last_valid_action
        elif self._assumed_opponent_action_on_nonactionable == "stand":
            return 0
        elif self._assumed_opponent_action_on_nonactionable == "none":
            return None
        else:
            raise RuntimeError("the value of 'assumed_opponent_action_on_nonactionable' is misconfigured! Should be either 'last', 'stand' or 'none'")

    def _select_apt_game_model(self, n: int) -> GameModel:
        raise NotImplementedError
        return self._game_model

    def append_info(self, p1_action: int, p2_actionable: bool):
        """Append an action that player 1 (the agent) performed."""
        self._past_agent_actions.append(p1_action)
        self._past_p2_actionables.append(p2_actionable)

    def reset(self):
        """Reset internal state, ideally done at the end of every episode."""
        self._past_agent_actions.clear()
        self._past_agent_actions.extend([0] * self._past_agent_actions.maxlen)
        self._past_p2_actionables.clear()
        self._past_p2_actionables.extend([0] * self._past_p2_actionables.maxlen)

    @property
    def assumed_opponent_action_on_nonactionable(self) -> Literal["str", "none", "stand"]:
        """The action that is assumed of the opponent when they can'torch.Tensor act."""
        return self._assumed_opponent_action_on_nonactionable

