import torch
from torch import nn
from torch.distributions import Categorical
from copy import deepcopy
from agents.action import ActionMap
from agents.base import FootsiesAgentTorch
from gymnasium import Env
from typing import Callable, Tuple
from agents.the_one.model import FullModel, RepresentationModule, AbstractGameModel, AbstractOpponentModel
from agents.the_one.reaction_time import ReactionTimeEmulator
from agents.a2c.agent import FootsiesAgent as A2CAgent
from agents.mimic.agent import PlayerModel
from footsies_gym.moves import FOOTSIES_MOVE_INDEX_TO_MOVE
from data import FootsiesDataset


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
        opponent_model_frameskip: bool = True,
        # Learning
        game_model_learning_rate: float = 1e-4,
        opponent_model_learning_rate: float = 1e-4,
    ):
        # Validate arguments
        if not over_simple_actions:
            raise NotImplementedError("non-simple actions are not yet supported")
        
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
        self.opponent_model_frameskip = opponent_model_frameskip

        # To report in the `model` property
        self.full_model = FullModel(
            game_model=self.game_model,
            opponent_model=self.opponent_model,
            actor=self.a2c.actor,
            critic=self.a2c.critic,
        )

        # Optimizers. The actor-critic module already takes care of its own optimizers.
        if self.game_model is not None:
            self.game_model_optimizer = torch.optim.SGD(self.game_model.parameters(), lr=game_model_learning_rate)
        if self.opponent_model is not None:
            self.opponent_model_optimizer = torch.optim.SGD(self.opponent_model.parameters(), lr=opponent_model_learning_rate)

        self.current_observation = None
        self.current_representation = None
        # In case simplified, temporally extended actions are being used. We need to keep track of them
        self.current_simple_action = None
        self.current_simple_action_frame = 0

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
        if len(self.a2c.actor.actor_layers) > 1:
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
        
        policy_parameters = dict(self.a2c.actor.actor_layers[0].named_parameters())
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
        if self.opponent_model is not None:
            predicted_opponent_action = self.opponent_model.predict(obs)
        else:
            predicted_opponent_action = None

        action = self.a2c.act(self.current_observation, info, predicted_opponent_action=predicted_opponent_action)
        return action

    def update(self, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict):
        # The actions are from info dictionary at the next step, but that's because it contains which action was performed in the *previous* step
        if self.over_simple_actions:
            agent_action = info["p1_move"]
            opponent_action = info["p2_move"]
            agent_action = ActionMap.simple_from_move_index(agent_action)
            opponent_action = ActionMap.simple_from_move_index(opponent_action)
        else:
            raise NotImplementedError("non-simple actions are not yet supported")
        agent_action_onehot = nn.functional.one_hot(torch.tensor(agent_action), num_classes=self.action_dim).unsqueeze(0)
        opponent_action_onehot = nn.functional.one_hot(torch.tensor(opponent_action), num_classes=self.opponent_action_dim).unsqueeze(0)

        # Update the different models.
        self.a2c.update(next_obs, reward, terminated, truncated, info)
        if self.game_model is not None:
            self._update_game_model(self.current_observation, agent_action_onehot, opponent_action_onehot, next_obs)
        if self.opponent_model is not None:
            self.opponent_model.update(self.current_observation, opponent_action_onehot)

    def _update_game_model(self, obs: torch.Tensor, agent_action_onehot: torch.Tensor, opponent_action_onehot: torch.Tensor, next_obs: torch.Tensor):
        """Calculate the game model loss, backpropagate and optimize"""
        if self.game_model is None:
            raise ValueError("agent wasn't instantiated with a game model, can't learn it")
        
        self.game_model_optimizer.zero_grad()

        with torch.no_grad():
            next_representation_target = self.representation(next_obs)

        next_representation_predicted = self.game_model(obs, agent_action_onehot, opponent_action_onehot)
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

    # NOTE: literally extracts the policy only, doesn't include any other component
    def extract_policy(self, env: Env) -> Callable[[dict], Tuple[bool, bool, bool]]:
        model = deepcopy(self.actor)

        def internal_policy(obs):
            probs = model(obs)
            return Categorical(probs=probs).sample().item()

        return super()._extract_policy(env, internal_policy)

    @property
    def model(self) -> nn.Module:
        return self.full_model
    
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
