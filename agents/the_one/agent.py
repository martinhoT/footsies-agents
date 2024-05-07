import torch as T
import logging
import random
import warnings
from torch import nn
from copy import deepcopy
from agents.base import FootsiesAgentOpponent
from gymnasium import Env
from agents.base import FootsiesAgentTorch
from agents.the_one.reaction_time import ReactionTimeEmulator, MultiStepPredictor
from agents.a2c.agent import A2CAgent
from agents.mimic.agent import MimicAgent
from agents.game_model.agent import GameModelAgent
from agents.wrappers import FootsiesSimpleActions
from time import process_time_ns

LOGGER = logging.getLogger("main.the_one")


class TheOneAgent(FootsiesAgentTorch):
    def __init__(
        self,
        # Dimensions
        obs_dim: int,
        action_dim: int,
        opponent_action_dim: int,
        # Modules
        a2c: A2CAgent,
        opponent_model: MimicAgent | None = None,
        game_model: GameModelAgent | None = None,
        reaction_time_emulator: ReactionTimeEmulator | None = None,
        # Modifiers
        rollback_as_opponent_model: bool = False,
        learn_a2c: bool = True,
        learn_opponent_model: bool = True,
        learn_game_model: bool = True,
    ):
        """
        FOOTSIES agent that integrates an opponent-aware reinforcement learning algorithm with an opponent model, along with reaction time.
        
        NOTE: if both a reaction time emulator and a game model are provided, then the game model will be implicitly used by the reaction time emulator
        to perform correction of delayed observations, which will cause a performance hit at inference time.

        Parameters
        ----------
        - `obs_dim`: the dimensionality of the observations
        - `action_dim`: the dimensionality of the actions
        - `opponent_action_dim`: the dimensionality of the opponent's actions
        - `a2c`: the opponent-aware reinforcement learning agent implementation
        - `opponent_model`: the opponent model, or `None` if one is not to be used
        - `game_model`: the game model, or `None` if one is not to be used
        - `reaction_time_emulator`: the reaction time emulator, or `None` if one is not to be used
        - `rollback_as_opponent_model`: whether to use rollback-based prediction as a stand-in for the opponent model.
        Only makes sense to be used if the opponent model is `None`
        - `game_model_learning_rate`: the learning rate of the player model
        - `learn_a2c`: whether to learn the A2C component at `update`
        - `learn_opponent_model`: whether to learn the opponent model component at `update`
        - `learn_game_model`: whether to learn the game model component at `update`
        
        Even if all of the `learn_...` arguments are `False`, `update` will still perform some necessary updates for acting (such as resetting state after episode termination/truncation)
        """
        # Validate arguments
        if rollback_as_opponent_model and opponent_model is not None:
            raise ValueError("can't have an opponent model when using the rollback strategy as an opponent predictor, can only use one or the other")
        if reaction_time_emulator is not None and opponent_model is None:
            raise ValueError("can't use reaction time without an opponent model, since reaction time depends on how well we model the opponent's behavior")
        if opponent_model is not None and opponent_model.p2_model is None:
            raise ValueError("specified an useless opponent model, it should have a model defined for player 2 (the opponent)")

        LOGGER.info("Agent was setup with opponent prediction strategy: %s", "rollback" if rollback_as_opponent_model else "opponent model" if opponent_model is not None else "random (unless doing curriculum learning)")

        # If we have both a reaction time emulator and a game model, add a predictor to correct delayed observations.
        if reaction_time_emulator is not None and game_model is not None:
            reaction_time_emulator.predictor = MultiStepPredictor(
                reaction_time_emulator,
                game_model,
            )

        # Store required values
        #  Dimensions
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.opponent_action_dim = opponent_action_dim
        #  Modules
        self.a2c = a2c
        self.gm = game_model
        self.opp = opponent_model
        self.reaction_time_emulator = reaction_time_emulator
        #  Modifiers
        self._rollback_as_opponent_model = rollback_as_opponent_model
        self._learn_a2c = learn_a2c
        self._learn_opponent_model = learn_opponent_model
        self._learn_game_model = learn_game_model

        # We set the previous opponent action to 0, which should correspond to a no-op (STAND).
        # This is only used for the rollback opponent prediction method.
        self._previous_valid_opponent_action = 0
        # Reaction time emulator can only be reset on environment reset, hence why we need to keep track of a flag.
        self._reaction_time_emulator_should_reset = True

        # Merely for tracking
        self._recently_predicted_opponent_action = None

        # Logging
        self._test_observations = None
        self._cumulative_reaction_time = 0
        self._cumulative_reaction_time_n = 0
        self._recent_act_elapsed_time_ns: int = 0
        self._recent_act_elapsed_time_ns_n: int = 0

        # Initial reset
        self.reset()

    # It's in this function that the current observation and representation variables are updated.
    # NOTE: during hitstop, the agent is acting in the first frame of hitstop; that means it does not matter how long the hitstop is, the agent won't be able to react to it
    @T.no_grad
    def act(self, obs: T.Tensor, info: dict) -> int:
        start_ns = process_time_ns()

        # Do some pre-processing.
        # Save the opponent's executed action. This is only really needed in case we are using the rollback prediction strategy.
        # Don't store None as the previous action, so that at the end of a temporal action we still have a reference to the action
        # that originated the temporal action in the first place.
        self._previous_valid_opponent_action = info["p2_simple"] if info["p2_was_actionable"] else self._previous_valid_opponent_action

        opponent_model_hidden_state = "auto"

        # Update the reaction time emulator and substitute obs with a perceived observation, which is delayed.
        if self.reaction_time_emulator is not None:
            if self._reaction_time_emulator_should_reset:
                self.reaction_time_emulator.reset(obs, info)
                self._reaction_time_emulator_should_reset = False
            
            self.reaction_time_emulator.register(obs, info)
        
        # Act, if that makes sense.
        if not info["p1_is_actionable"] or not info["agent_simple_completed"]:
            return 0
        
        # Correct the observation if needed.
        # Note that we need to have the up-to-date opponent model hidden state.
        if self.reaction_time_emulator is not None:
            predicted_obs, reaction_time, opponent_model_hidden_state = self.reaction_time_emulator.react
            self._cumulative_reaction_time += reaction_time
            self._cumulative_reaction_time_n += 1
            
            obs = predicted_obs

        # Predict an action for the opponent.
        # NOTE: this doesn't take into account whether the opponent is actually able to act or not
        if self.opp is not None and self.opp.p2_model is not None:
            predicted_opponent_distribution, _ = self.opp.p2_model.network.distribution(obs, opponent_model_hidden_state)
            predicted_opponent_action = int(predicted_opponent_distribution.sample().item())
        elif self._rollback_as_opponent_model:
            predicted_opponent_action = self._previous_valid_opponent_action
        else:
            opponent_policy = info.get("next_opponent_policy", None)
            if opponent_policy is not None:
                predicted_opponent_action = random.choices(range(self.opponent_action_dim), weights=opponent_policy, k=1)[0]
            else:
                predicted_opponent_action = None
        
        # In case the opponent's considered action dimensionality is actually smaller
        if predicted_opponent_action is not None:
            predicted_opponent_action = min(predicted_opponent_action, self.opponent_action_dim - 1)

        action = self.a2c.act(obs, info, predicted_opponent_action=predicted_opponent_action)
        self._recently_predicted_opponent_action = predicted_opponent_action

        self._recent_act_elapsed_time_ns += process_time_ns() - start_ns
        self._recent_act_elapsed_time_ns_n += 1
        return action

    # NOTE: reaction time is not used here, it's only used for acting not for learning.
    #       We aren't considering delayed information for the updates, since neural network updates are slow and thus
    #       it's unlikely that these privileged updates are going to make a difference. In turn, the code is simpler.
    def update(self, obs: T.Tensor, next_obs: T.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict):
        # Update the different models
        if self._learn_a2c or self._learn_game_model or self._learn_opponent_model:
            if self.opp is not None and self.opp.p2_model is not None:
                if "next_opponent_policy" in next_info:
                    warnings.warn("The 'next_opponent_policy' was already provided in info dictionary, but will be overwritten with the opponent model.")
                next_opponent_policy, _ = self.opp.p2_model.network.probabilities(next_obs, "auto")
                next_info["next_opponent_policy"] = next_opponent_policy.detach().squeeze(0)

            # Cap the opponent's action in case the action dimensionality we consider is smaller
            # (for instance, if it's 1 which is as if we did not consider them at all)
            if info["p2_simple"] is not None:
                info["p2_simple"] = min(info["p2_simple"], self.opponent_action_dim - 1)
            if next_info["p2_simple"] is not None:
                next_info["p2_simple"] = min(next_info["p2_simple"], self.opponent_action_dim - 1)

            if self._learn_a2c:
                self.a2c.update(obs, next_obs, reward, terminated, truncated, info, next_info)
            if self._learn_game_model and self.gm is not None:
                self.gm.update(obs, next_obs, reward, terminated, truncated, info, next_info)
            if self._learn_opponent_model and self.opp is not None:
                self.opp.update(obs, next_obs, reward, terminated, truncated, info, next_info)

        # The reaction time emulator needs to know when an episode terminated/truncated, so we reset (which includes resetting the reaction time emulator).
        if terminated or truncated:
            self.reset()

    def preprocess(self, env: Env):
        # Obtain the way the environment is resolving opponent actions when it can't act.
        # This is important when we are doing the multi-step prediction ourselves.
        e = env
        assumed_opponent_action_on_nonactionable = None
        while e != e.unwrapped:
            if isinstance(e, FootsiesSimpleActions):
                assumed_opponent_action_on_nonactionable = e.assumed_opponent_action_on_nonactionable
                break
            e = e.env # type: ignore
        
        if assumed_opponent_action_on_nonactionable is None:
            raise ValueError("expected environment to have the `FootsiesSimpleActions` wrapper with `assumed_opponent_action_on_nonactionable` property set, but that's not the case")
    
        if self.reaction_time_emulator is not None and self.reaction_time_emulator.predictor is not None:
            self.reaction_time_emulator.predictor.assumed_opponent_action_on_nonactionable = assumed_opponent_action_on_nonactionable

    def reset(self):
        self._reaction_time_emulator_should_reset = True
        self._previous_valid_opponent_action = 0

    # NOTE: watch out! In the current way we handle the SimpleActions wrapper we don't include some keys in info that allow for the A2C module to learn.
    #       As long as the opponent doesn't learn, and no other component uses that information, there should be no problem.
    def extract_opponent(self, env: Env) -> FootsiesAgentOpponent:
        opponent = deepcopy(self)
        opponent.learn_a2c = False
        opponent.learn_opponent_model = False
        opponent.learn_game_model = False

        return FootsiesAgentOpponent(
            agent=opponent,
            env=env,
        )

    # Only the actor-critic modules are shared in Hogwild!, not the game model or opponent model
    @property
    def shareable_model(self) -> nn.Module:
        return self.a2c.model

    @property
    def recently_predicted_opponent_action(self) -> int | None:
        """The most recent prediction for the opponent's action done in the last `act` call."""
        return self._recently_predicted_opponent_action

    @property
    def learn_a2c(self) -> bool:
        """Whether the agent is learning the A2C component at `update`."""
        return self._learn

    @learn_a2c.setter
    def learn_a2c(self, value: bool):
        self._learn = value

    @property
    def learn_opponent_model(self) -> bool:
        """Whether the agent is learning the opponent model at `update`."""
        return self._learn_opponent_model

    @learn_opponent_model.setter
    def learn_opponent_model(self, value: bool):
        self._learn_opponent_model = value

    @property
    def learn_game_model(self) -> bool:
        """Whether the agent is learning the game model at `update`."""
        return self._learn_game_model
    
    @learn_game_model.setter
    def learn_game_model(self, value: bool):
        self._learn_game_model = value

    def load(self, folder_path: str):
        # Load actor-critic
        try:
            self.a2c.load(folder_path)
        except FileNotFoundError:
            LOGGER.warning("Could not find the A2C model to load, will use one from scratch")

        # Load game model
        if self.gm is not None:
            try:
                self.gm.load(folder_path)
            except FileNotFoundError:
                LOGGER.warning("Could not find game model to load, will use one from scratch")

        # Load opponent model (even though this is meant to be discardable)
        if self.opp is not None:
            try:
                self.opp.load(folder_path)
            except FileNotFoundError:
                LOGGER.warning("Could not find opponent model to load, will use one from scratch")

    def save(self, folder_path: str):
        # Save actor-critic
        self.a2c.save(folder_path)

        # Save game model
        if self.gm is not None:
            self.gm.save(folder_path)
        
        # Save opponent model (even though this is meant to be discardable)
        if self.opp is not None:
            self.opp.save(folder_path)

    def evaluate_average_reaction_time(self) -> float | None:
        res = (
            self._cumulative_reaction_time / self._cumulative_reaction_time_n
        ) if self._cumulative_reaction_time_n != 0 else None

        self._cumulative_reaction_time = 0
        self._cumulative_reaction_time_n = 0

        return res
    
    def evaluate_act_elapsed_time_seconds(self) -> float | None:
        res = (
            self._recent_act_elapsed_time_ns / (1e9 * self._recent_act_elapsed_time_ns_n)
        ) if self._recent_act_elapsed_time_ns_n != 0 else None

        self._recent_act_elapsed_time_ns = 0
        self._recent_act_elapsed_time_ns_n = 0

        return res
