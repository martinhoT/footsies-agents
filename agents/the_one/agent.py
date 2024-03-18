import os
import torch
import logging
import random
from torch import nn
from agents.action import ActionMap
from agents.base import FootsiesAgentTorch
from gymnasium import Env
from typing import Callable, Tuple
from agents.the_one.model import FullModel, RepresentationModule, AbstractGameModel
from agents.the_one.reaction_time import ReactionTimeEmulator
from agents.a2c.agent import FootsiesAgent as A2CAgent
from agents.a2c.a2c import ActorNetwork
from agents.mimic.agent import PlayerModel
from agents.torch_utils import observation_invert_perspective_flattened
from data import FootsiesDataset

LOGGER = logging.getLogger("main.the_one")


# TODO: use reaction time emulator
class FootsiesAgent(FootsiesAgentTorch):
    def __init__(
        self,
        # Dimensions
        obs_dim: int,
        action_dim: int,
        opponent_action_dim: int,
        # Modules
        representation: RepresentationModule,
        a2c: A2CAgent,
        opponent_model: PlayerModel = None,
        game_model: AbstractGameModel = None,
        reaction_time_emulator: ReactionTimeEmulator = None,
        # Modifiers
        over_simple_actions: bool = False,
        remove_special_moves: bool = False,
        rollback_as_opponent_model: bool = False,
        # Learning
        game_model_learning_rate: float = 1e-4,
    ):
        # Validate arguments
        if not over_simple_actions:
            raise NotImplementedError("non-simple actions are not yet supported")
        if rollback_as_opponent_model and opponent_model is not None:
            raise ValueError("can't have an opponent model when using the rollback strategy as an opponent predictor, can only use one or the other")

        # Store required values
        #  Dimensions
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.opponent_action_dim = opponent_action_dim
        #  Modules
        self.representation = representation
        self.a2c = a2c
        self.game_model = game_model
        self.opponent_model = opponent_model
        self.reaction_time_emulator = reaction_time_emulator
        #  Modifiers
        self.over_simple_actions = over_simple_actions
        self.remove_agent_special_moves = remove_special_moves
        self.rollback_as_opponent_model = rollback_as_opponent_model

        # To report in the `model` property
        self._full_model = FullModel(
            game_model=self.game_model,
            opponent_model=None if self.opponent_model is None else self.opponent_model.network,
            actor_critic=self.a2c.model,
        )

        # Optimizers. The actor-critic module already takes care of its own optimizers.
        if self.game_model is not None:
            self.game_model_optimizer = torch.optim.SGD(self.game_model.parameters(), lr=game_model_learning_rate)

        self.current_observation = None
        self.current_info = None
        self.current_representation = None
        # In case simplified, temporally extended actions are being used. We need to keep track of them
        self.current_simple_action = None
        self.current_simple_action_frame = 0
        # We set the previous opponent action to 0, which should correspond to a no-op (STAND)
        self.previous_valid_opponent_action = 0

        # Merely for tracking
        self._recently_predicted_opponent_action = None

        # Loss trackers
        self.cumulative_loss_game_model = 0
        self.cumulative_loss_game_model_n = 0
        self.cumulative_loss_opponent_model = 0
        self.cumulative_loss_opponent_model_n = 0

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
        if len(self.game_model.game_model_layers) > 1:
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
        game_model_parameters = dict(self.game_model.game_model_layers[0].named_parameters())
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
        self.current_observation = obs
        self.current_info = info
        if self.opponent_model is not None:
            predicted_opponent_action = self.opponent_model.predict(obs)
        elif self.rollback_as_opponent_model:
            predicted_opponent_action = self.previous_valid_opponent_action
        else:
            opponent_policy = info.get("next_opponent_policy", None)
            if opponent_policy is not None:
                predicted_opponent_action = random.choices(range(self.opponent_action_dim), weights=opponent_policy, k=1)[0]
            else:
                predicted_opponent_action = None

        action = self.a2c.act(self.current_observation, info, predicted_opponent_action=predicted_opponent_action)
        self._recently_predicted_opponent_action = predicted_opponent_action
        return action

    def update(self, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict):
        # Get the actions that were effectively performed by each player on the previous step
        if self.over_simple_actions:
            agent_action, opponent_action = ActionMap.simples_from_transition_ori(self.current_info, info)
            if self.remove_agent_special_moves:
                # Convert the detected special move input (how did it even happen??) to a simple action
                if agent_action == 8 or agent_action == 7:
                    LOGGER.warning("We detected the agent performing a special move, even though they can't perform special moves! Will convert to the respective attack action.\nCurrent observation: %s\nNext observation: %s", self.current_observation, next_obs)
                    agent_action -= 2
        else:
            raise NotImplementedError("non-simple actions are not yet supported")

        # Update the different models
        self.a2c.update(next_obs, reward, terminated, truncated, info)
        if self.game_model is not None:
            self._update_game_model(self.current_observation, agent_action, opponent_action, next_obs)
        if self.opponent_model is not None and opponent_action is not None:
            self.opponent_model.update(self.current_observation, opponent_action)

        # Save the opponent's executed action. This is only really needed in case we are using the rollback prediction strategy.
        # Don't store None as the previous action, so that at the end of a temporal action we still have a reference to the action
        # that originated the temporal action in the first place. This is important for the rollback-based prediction strategy.
        # This variable is only used for the rollback-based prediction anyway.
        self.previous_valid_opponent_action = opponent_action if opponent_action is not None else self.previous_valid_opponent_action

    def _update_game_model(self, obs: torch.Tensor, agent_action: int, opponent_action: int, next_obs: torch.Tensor):
        """Calculate the game model loss, backpropagate and optimize"""
        if self.game_model is None:
            raise ValueError("agent wasn't instantiated with a game model, can't learn it")
        
        self.game_model_optimizer.zero_grad()

        with torch.no_grad():
            next_representation_target = self.representation(next_obs)

        next_representation_predicted = self.game_model(obs, agent_action, opponent_action)
        game_model_loss = torch.nn.functional.mse_loss(next_representation_predicted, next_representation_target)
        game_model_loss.backward()

        self.game_model_optimizer.step()

        self.cumulative_loss_game_model += game_model_loss.item()
        self.cumulative_loss_game_model_n += 1

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
        actor = self.a2c._actor.clone()

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
                    _, opp_action = ActionMap.simples_from_torch_transition(actor.previous_observation__, obs)
                
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
        if self.game_model is not None:
            game_model_path = os.path.join(folder_path, "game_model")
            self.game_model.load_state_dict(torch.load(game_model_path))

        # Load opponent model (even though this is meant to be discardable)
        if self.opponent_model is not None:
            opponent_model_path = os.path.join(folder_path, "opponent_model")
            self.opponent_model.load(opponent_model_path)

    def save(self, folder_path: str):
        # Save actor-critic
        self.a2c.save(folder_path)

        # Save game model
        if self.game_model is not None:
            game_model_path = os.path.join(folder_path, "game_model")
            torch.save(self.game_model.state_dict(), game_model_path)
        
        # Save opponent model (even though this is meant to be discardable)
        if self.opponent_model is not None:
            opponent_model_path = os.path.join(folder_path, "opponent_model")
            self.opponent_model.save(opponent_model_path)

    def evaluate_average_loss_game_model(self) -> float:
        res = (
            self.cumulative_loss_game_model / self.cumulative_loss_game_model_n
        ) if self.cumulative_loss_game_model_n != 0 else 0

        self.cumulative_loss_game_model = 0
        self.cumulative_loss_game_model_n = 0

        return res

    def evaluate_average_loss_opponent_model(self) -> float:
        res = (
            self.cumulative_loss_opponent_model / self.cumulative_loss_opponent_model_n
        ) if self.cumulative_loss_opponent_model_n != 0 else 0

        self.cumulative_loss_opponent_model = 0
        self.cumulative_loss_opponent_model_n = 0

        return res
