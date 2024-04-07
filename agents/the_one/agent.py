import os
import torch
import logging
import random
import warnings
from torch import nn
from agents.action import ActionMap
from agents.base import FootsiesAgentTorch
from gymnasium import Env
from typing import Callable, Tuple
from agents.the_one.model import FullModel, RepresentationModule
from agents.the_one.reaction_time import ReactionTimeEmulator
from agents.a2c.agent import A2CAgent
from agents.a2c.a2c import ActorNetwork
from agents.mimic.agent import MimicAgent
from agents.game_model.agent import GameModelAgent
from agents.torch_utils import observation_invert_perspective_flattened
from collections import deque
from data import FootsiesDataset

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
        representation: RepresentationModule | None = None,
        opponent_model: MimicAgent | None = None,
        game_model: GameModelAgent | None = None,
        reaction_time_emulator: ReactionTimeEmulator | None = None,
        # Modifiers
        remove_special_moves: bool = False,
        rollback_as_opponent_model: bool = False,
    ):
        """
        FOOTSIES agent that integrates an opponent-aware reinforcement learning algorithm with an opponent model, along with reaction time.
        
        Parameters
        ----------
        - `obs_dim`: the dimensionality of the observations
        - `action_dim`: the dimensionality of the actions
        - `opponent_action_dim`: the dimensionality of the opponent's actions
        - `a2c`: the opponent-aware reinforcement learning agent implementation
        - `representation`: a representation network that is shared among various networks, or `None` if one is not used
        - `opponent_model`: the opponent model, or `None` if one is not to be used
        - `game_model`: the game model, or `None` if one is not to be used
        - `reaction_time_emulator`: the reaction time emulator, or `None` if one is not to be used
        - `remove_special_moves`: whether to explicitly not consider special moves as part of the agent's action space
        - `rollback_as_opponent_model`: whether to use rollback-based prediction as a stand-in for the opponent model.
        Only makes sense to be used if the opponent model is `None`
        - `game_model_learning_rate`: the learning rate of the player model
        """
        # Validate arguments
        if rollback_as_opponent_model and opponent_model is not None:
            raise ValueError("can't have an opponent model when using the rollback strategy as an opponent predictor, can only use one or the other")
        if reaction_time_emulator is not None and opponent_model is None:
            raise ValueError("can't use reaction time without an opponent model, since reaction time depends on how well we model the opponent's behavior")
        if reaction_time_emulator is not None and game_model is None:
            raise ValueError("can't use reaction time without a game model, since prediction of the actual current observation depends on the game model")

        LOGGER.info("Agent was setup with opponent prediction strategy: %s", "rollback" if rollback_as_opponent_model else "opponent model" if opponent_model is not None else "random (unless doing curriculum learning)")

        # Store required values
        #  Dimensions
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.opponent_action_dim = opponent_action_dim
        #  Modules
        self.representation = representation
        self.a2c = a2c
        self.env_model = game_model
        self.opponent_model = opponent_model
        self.reaction_time_emulator = reaction_time_emulator
        #  Modifiers
        self.remove_agent_special_moves = remove_special_moves
        self.rollback_as_opponent_model = rollback_as_opponent_model

        # To report in the `model` property
        self._full_model = FullModel(
            game_model=self.env_model,
            opponent_model=None if self.opponent_model is None else self.opponent_model.p2_model.network,
            actor_critic=self.a2c.model,
        )

        # The perceived observation, delayed N time steps according to reaction time.
        self._current_observation_delayed = None
        # We set the previous opponent action to 0, which should correspond to a no-op (STAND).
        # This is only used for the rollback opponent prediction method.
        self._previous_valid_opponent_action = 0
        # The hidden state of the opponent model (only matters if recurrent).
        # This version of the hidden state is the one calculated after advancing the opponent model up to predicted observation.
        self._current_opponent_model_hidden_state_perceived = None
        # The actions that the agent has previously performed.
        # This is a buffer of the same size of the reaction time observation buffer to aid in multi-step prediction of the current state.
        # The buffer doesn't need to include the action that was performed at the current state, so we subtract 1.
        past_agent_actions_size = (reaction_time_emulator.history_size - 1) if reaction_time_emulator is not None else 0
        self._past_agent_actions = deque([], maxlen=past_agent_actions_size)

        # Merely for tracking
        self._recently_predicted_opponent_action = None

        # Logging
        self._test_observations = None
        self._cumulative_reaction_time = 0
        self._cumulative_reaction_time_n = 0

    def env_concat(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Obtain the concatenated weights that calculate the next environment observation `n` steps into the future.
        Only works if the game model, opponent model and agent policy are linear.

        Parameters
        ----------
        - `n`: the number of steps to predict

        Returns
        -------
        - `mtx`: the weight matrix to multiply with the observation `obs` (or the representation if one is used)
        - `bias`: the bias to add

        The final step is computed as `mtx @ obs + bias`
        """
        raise NotImplementedError("not supported")
    
        if len(self.env_model.game_model_layers) > 1:
            raise ValueError("the game model must be linear to use this method")
        if len(self.opponent_model.opponent_model_layers) > 1:
            raise ValueError("the opponent model must be linear to use this method")
        if len(self.a2c._actor.actor_layers) > 1:
            raise ValueError("the actor must be linear to use this method")

        obs_dim = self.representation.representation_dim if self.representation is not None else self.obs_dim
        action_dim = self.action_dim
        opponent_action_dim = self.opponent_action_dim

        # Matrices for implementing the concatenation of input vectors
        C_s = torch.zeros((obs_dim + action_dim + opponent_action_dim, obs_dim), dtype=torch.float32)
        C_a = torch.zeros((obs_dim + action_dim + opponent_action_dim, action_dim), dtype=torch.float32)
        C_o = torch.zeros((obs_dim + action_dim + opponent_action_dim, opponent_action_dim), dtype=torch.float32)

        C_s[:obs_dim, :] = torch.eye(obs_dim)
        C_a[obs_dim:(obs_dim + action_dim), :] = torch.eye(action_dim)
        C_o[(obs_dim + action_dim):(obs_dim + action_dim + opponent_action_dim), :] = torch.eye(opponent_action_dim)

        # Game model, opponent model and agent policy parameters.
        # Make the bias vectors column vectors to make sure operations are done correctly.
        game_model_parameters = dict(self.env_model.game_model_layers[0].named_parameters())
        W_g = game_model_parameters["weight"].data
        b_g = game_model_parameters["bias"].data.unsqueeze(1)
        
        opponent_model_parameters = dict(self.opponent_model.opponent_model_layers[0].named_parameters())
        W_o = opponent_model_parameters["weight"].data
        b_o = opponent_model_parameters["bias"].data.unsqueeze(1)
        
        policy_parameters = dict(self.a2c._actor.actor_layers[0].named_parameters())
        W_a = policy_parameters["weight"].data
        b_a = policy_parameters["bias"].data.unsqueeze(1)
        
        X = W_g @ C_s + W_g @ C_a @ W_a + W_g @ C_o @ W_o

        mtx = torch.linalg.matrix_power(X, n)

        bias = (
              sum(torch.linalg.matrix_power(X, i) for i in range(n)) @ W_g @ C_a @ b_a
            + sum(torch.linalg.matrix_power(X, i) for i in range(n)) @ W_g @ C_o @ b_o
            + sum(torch.linalg.matrix_power(X, i) for i in range(n)) @ b_g
        )

        return mtx, bias

    # It's in this function that the current observation and representation variables are updated
    def act(self, obs: torch.Tensor, info: dict) -> int:
        # Update the reaction time emulator and substitute obs with a perceived observation, which is delayed.
        if self.reaction_time_emulator is not None:
            if self._current_observation_delayed is None:
                self.reaction_time_emulator.fill_history(obs)
                self._current_observation_delayed = obs
            
            opponent_probabilities, hidden_state = self.opponent_model.p2_model.network.probabilities(self._current_observation_delayed, self._current_opponent_model_hidden_state_perceived)
            opponent_probabilities = opponent_probabilities.detach()
            self._current_opponent_model_hidden_state_perceived = hidden_state.detach()
            decision_distribution = self.a2c.learner.actor.decision_distribution(self._current_observation_delayed, opponent_probabilities, detached=True)
            decision_entropy = decision_distribution.entropy().item()
            self._current_observation_delayed, reaction_time, skipped_observations = self.reaction_time_emulator.register_and_perceive(obs, decision_entropy)
            for skipped in skipped_observations:
                if ActionMap.is_state_actionable_torch(skipped, False):
                    _, self._current_opponent_model_hidden_state_perceived = self.opponent_model.p2_model.network.compute_hidden_state(skipped, self._current_opponent_model_hidden_state_perceived)

            obs, opponent_model_hidden_state = self.multi_step_prediction(self._current_observation_delayed, reaction_time, self._current_opponent_model_hidden_state_perceived)
            
            LOGGER.debug("Reacted with decision distribution %s and entropy %s, resulting in a reaction time of %s, and predicted the current observation is %s", decision_distribution.probs, decision_entropy, reaction_time, obs)
            self._cumulative_reaction_time += reaction_time
            self._cumulative_reaction_time_n += 1
            
        if self.opponent_model is not None:
            predicted_opponent_distribution, _ = self.opponent_model.p2_model.network.distribution(obs, opponent_model_hidden_state)
            predicted_opponent_action = predicted_opponent_distribution.sample()
        elif self.rollback_as_opponent_model:
            predicted_opponent_action = self._previous_valid_opponent_action
        else:
            opponent_policy = info.get("next_opponent_policy", None)
            if opponent_policy is not None:
                predicted_opponent_action = random.choices(range(self.opponent_action_dim), weights=opponent_policy, k=1)[0]
            else:
                predicted_opponent_action = None

        action = self.a2c.act(obs, info, predicted_opponent_action=predicted_opponent_action)
        self._recently_predicted_opponent_action = predicted_opponent_action
        # We need to store the simple action, not the discretized one!
        self._past_agent_actions.append(self.a2c.current_action)
        return action

    # NOTE: reaction time is not used here, it's only used for acting not for learning.
    #       We aren't considering delayed information for the updates, since neural network updates are slow and thus
    #       it's unlikely that these privileged updates are going to make a difference. In turn, the code is simpler.
    def update(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict):
        # Get the actions that were effectively performed by each player on the previous step
        agent_action, opponent_action = ActionMap.simples_from_transition_ori(info, next_info)
        if self.remove_agent_special_moves:
            # Convert the detected special move input to a simple action
            if agent_action == 8 or agent_action == 7:
                LOGGER.warning("We detected the agent performing a special move, even though they can't perform special moves! Will convert to the respective attack action.\nCurrent observation: %s\nNext observation: %s", self.current_observation, next_obs)
                agent_action -= 2

        if self.opponent_model is not None:
            if "next_opponent_policy" in next_info:
                warnings.warn("The 'next_opponent_policy' was already provided in info dictionary, but will be overwritten with the opponent model.")
            next_opponent_policy, _ = self.opponent_model.p2_model.network.probabilities(next_obs, "auto")
            next_info["next_opponent_policy"] = next_opponent_policy.detach().squeeze()

        # Update the different models
        self.a2c.update(obs, next_obs, reward, terminated, truncated, info, next_info)
        if self.env_model is not None:
            self.env_model.update_with_simple_actions(obs, agent_action, opponent_action, next_obs)
        if self.opponent_model is not None:
            self.opponent_model.update_with_simple_actions(obs, None, opponent_action, terminated or truncated)

        # Save the opponent's executed action. This is only really needed in case we are using the rollback prediction strategy.
        # Don't store None as the previous action, so that at the end of a temporal action we still have a reference to the action
        # that originated the temporal action in the first place.
        self._previous_valid_opponent_action = opponent_action if opponent_action is not None else self._previous_valid_opponent_action

        # The reaction time emulator needs to know when an episode terminated/truncated.
        if terminated or truncated:
            self._current_observation_delayed = None
    
    def multi_step_prediction(self, obs: torch.Tensor, n: int, opponent_model_hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the observation `n` time steps into the future, using the agent's past executed actions, opponent model and game model."""
        o = self.env_model.game_model.preprocess_observation(obs)
        for t in reversed(range(n)):
            p2_actionable = ActionMap.is_state_actionable_torch(o, False)

            # TODO: support "last" action assumption, like in the critic
            if p2_actionable:
                p2_action, opponent_model_hidden_state = self.opponent_model.p2_model.network.probabilities(o, opponent_model_hidden_state)
                p2_action = p2_action.detach()
                opponent_model_hidden_state = opponent_model_hidden_state.detach()
            else:
                p2_action = torch.zeros(1, self.opponent_action_dim)
                p2_action[0, 0] = 1.0

            p1_action = self._past_agent_actions[-1 - t]
            
            o = self.env_model.game_model.network(o, p1_action, p2_action).detach()
        
        final_obs = self.env_model.game_model.postprocess_prediction(o)
        return final_obs, opponent_model_hidden_state

    def initialize(
        self,
        dataset: FootsiesDataset,
        policy: bool = True,
        value: bool = False,
        game_model: bool = False,
        opponent_model: bool = False,
        frozen_representation: bool = False,
        agent_is_p1: bool = True,
    ):
        """
        Initialize models from a pre-built dataset of episodes.
        
        By default, only initializes the policy, which is what one would do in imitation learning.
        This however has implications if the different models share a common representation, as the representation
        will be skewed to favour the policy, unless `frozen_representation` is set to `True`.
        
        Parameters
        ----------
        - `dataset`: the dataset of episodes
        - `policy`: whether to update the policy
        - `value`: whether to update the value function
        - `game_model`: whether to update the game model
        - `opponent_model`: whether to update the opponent model 
        - `frozen_representation`: whether to freeze the representation module
        - `agent_is_p1`: whether player 1 is to be treated as the agent. If `False`, player 2 is treated as the agent
        """
        if value or game_model or opponent_model:
            raise NotImplementedError("initializing anything, but the policy, is not implemented yet")
        
        if policy:
            for episode in dataset.episodes:
                for obs, next_obs, reward, p1_action, p2_action in episode:
                    action = p1_action if agent_is_p1 else p2_action
                    self.imitator.learn(obs, action, frozen_representation)

    # NOTE: extracts the policy only, doesn't include any other component
    def extract_policy(self, env: Env) -> Callable[[dict], Tuple[bool, bool, bool]]:
        # We assume the actor is going to be used as an opponent
        actor = self.a2c.learner.actor.clone(p1=False)

        class ExtractedPolicy:
            def __init__(self, actor: ActorNetwork, deterministic: bool = False):
                self.actor = actor
                self.deterministic = deterministic
                self.current_action = None
                self.current_action_iterator = None
            
            def __call__(self, obs) -> int:
                # Invert the perspective, since the agent was trained as if they were on the left side of the screen
                obs = observation_invert_perspective_flattened(obs)

                opp_action = None
                previous_observation = getattr(actor, "previous_observation__") if hasattr(actor, "previous_observation__") else None
                if previous_observation is not None:
                    # Weak, rollback-inspired opponent model: if the opponent did action X, then they will keep doing X
                    _, opp_action = ActionMap.simples_from_transition_torch(actor.previous_observation__, obs)
                
                # When in doubt, assume the opponent is standing
                if opp_action is None:
                    opp_action = 0

                # Keep sampling the next discrete action from the simple action
                if self.current_action is not None:
                    try:
                        action = next(self.current_action_iterator)
                
                    except StopIteration:
                        self.current_action = None
                
                if self.current_action is None:
                    # Sample the action
                    if self.deterministic:
                        self.current_action = actor.probabilities(obs, next_opponent_action=opp_action).argmax().item()
                    else:
                        self.current_action = actor.sample_action(obs, next_opponent_action=opp_action).item()
                    self.current_action_iterator = iter(ActionMap.simple_to_discrete(self.current_action))
                    action = next(self.current_action_iterator)

                # Save the current observation in an attribute that *surely* won't overlap with an existing one
                actor.previous_observation__ = obs
                
                # We need to invert the action since the agent was trained to perform actions as if they were on the left side of the screen,
                # so we need to mirror them
                return ActionMap.invert_discrete(action)

        policy = ExtractedPolicy(actor)

        return super()._extract_policy(env, policy)

    @property
    def model(self) -> nn.Module:
        return self._full_model
    
    # Only the actor-critic modules are shared in Hogwild!, not the game model or opponent model
    @property
    def shareable_model(self) -> nn.Module:
        return self.a2c.model

    @property
    def recently_predicted_opponent_action(self) -> int:
        """The most recent prediction for the opponent's action done in the last `act` call."""
        return self._recently_predicted_opponent_action

    def load(self, folder_path: str):
        # Load actor-critic
        self.a2c.load(folder_path)

        # Load game model
        if self.env_model is not None:
            game_model_path = os.path.join(folder_path, "game_model")
            self.env_model.load_state_dict(torch.load(game_model_path))

        # Load opponent model (even though this is meant to be discardable)
        if self.opponent_model is not None:
            self.opponent_model.load(folder_path)

    def save(self, folder_path: str):
        # Save actor-critic
        self.a2c.save(folder_path)

        # Save game model
        if self.env_model is not None:
            game_model_path = os.path.join(folder_path, "game_model")
            torch.save(self.env_model.state_dict(), game_model_path)
        
        # Save opponent model (even though this is meant to be discardable)
        if self.opponent_model is not None:
            self.opponent_model.save(folder_path)

    def evaluate_average_reaction_time(self) -> float:
        res = (
            self._cumulative_reaction_time / self._cumulative_reaction_time_n
        ) if self._cumulative_reaction_time_n != 0 else 0

        self._cumulative_reaction_time = 0
        self._cumulative_reaction_time_n = 0

        return res
