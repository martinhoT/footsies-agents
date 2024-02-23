import numpy as np
import torch
from torch import nn
from os import path
from agents.action import ActionMap
from agents.base import FootsiesAgentTorch
from gymnasium import Env
from typing import Callable, Tuple
from agents.the_one.model import FullModel, RepresentationModule, AbstractGameModel, AbstractOpponentModel
from agents.the_one.reaction_time import ReactionTimeEmulator
from agents.a2c.a2c import A2CModule, ActorNetwork, CriticNetwork
from footsies_gym.moves import FOOTSIES_MOVE_INDEX_TO_MOVE


# TODO: use reaction time emulator
class FootsiesAgent(FootsiesAgentTorch):
    def __init__(
        self,
        # Dimensions
        observation_space_size: int,
        action_space_size: int,
        representation_dim: int,
        # Representation module structure
        representation_hidden_layer_sizes_specification: str = "",
        representation_hidden_layer_activation_specification: nn.Module = nn.Identity,
        # Modifiers
        consider_actions_in_representation: bool = False,
        over_primitive_actions: bool = False,
        game_model_double_grad: bool = False,   # let gradients flow both considering the prediction and target, and not just prediction (i.e. target is not detached)
        opponent_model_learn: bool = False,
        actor_critic_frameskip: bool = True,
        opponent_model_frameskip: bool = True,
        # Learning
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        game_model_learning_rate: float = 1e-4,
        opponent_model_learning_rate: float = 1e-4,
        # Other modules
        reaction_time_emulator: ReactionTimeEmulator = None,
        #  these will only be applied if the `reaction_time_emulator` argument is None
        reaction_time_emulator_minimum: int = 15,
        reaction_time_emulator_maximum: int = 29,
        # Keyword arguments for modules. Accepts:
        # - `actor_critic.{kwarg}`
        # - `reaction_time_emulator.{kwarg}`
        **kwargs,
    ):
        # Validate arguments
        if over_primitive_actions:
            raise NotImplementedError("primitive actions are not yet supported")
        
        if game_model_double_grad:
            raise NotImplementedError("backpropagation through targets for the game model is not (yet?) supported")

        unknown_kwargs = {k for k in kwargs.keys() if (".") not in k}
        if unknown_kwargs:
            raise ValueError(f"module keyword arguments must be prefixed with the module name, e.g. `actor_critic.` or `reaction_time_emulator.`, unknown: {unknown_kwargs}")

        actor_critic_kwargs = {k.split(".")[1]: v for k, v in kwargs.items() if k.startswith("actor_critic.")}
        reaction_time_emulator_kwargs = {k.split(".")[1]: v for k, v in kwargs.items() if k.startswith("reaction_time_emulator.")}
        unknown_module_kwargs = {k.split(".")[0] for k in kwargs.keys()} - {"actor_critic", "reaction_time_emulator"}
 
        if unknown_module_kwargs:
            raise ValueError(f"unknown module keyword arguments: {unknown_module_kwargs}")
        
        # Populate kwargs with default values
        actor_critic_kwargs = {
            "discount": 1.0,
            "actor_learning_rate": 1e-4,
            "critic_learning_rate": 1e-4,
            "actor_eligibility_traces_decay": 0.0,
            "critic_eligibility_traces_decay": 0.0,
            "actor_entropy_loss_coef": 0.0,
            **actor_critic_kwargs,
        }

        reaction_time_emulator_kwargs = {
            "inaction_probability": 0.0,
            "multiplier": 1.0,
            "additive": 0.0,
            "history_size": 30,
            **reaction_time_emulator_kwargs,
        }

        # Initialize necessary values according to passed arguments
        obs_dim = observation_space_size
        action_dim = action_space_size if over_primitive_actions else ActionMap.n_simple()
        opponent_action_dim = action_space_size if over_primitive_actions else ActionMap.n_simple()

        if reaction_time_emulator is None:
            reaction_time_emulator = ReactionTimeEmulator(**reaction_time_emulator_kwargs)
            reaction_time_emulator.confine_to_range(reaction_time_emulator_minimum, reaction_time_emulator_maximum, action_dim)

        # Store required values
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.opponent_action_dim = opponent_action_dim
        self.representation_dim = representation_dim
        self.game_model_double_grad = game_model_double_grad
        self.over_primitive_actions = over_primitive_actions
        self.opponent_model_learn = opponent_model_learn
        self.actor_critic_frameskip = actor_critic_frameskip
        self.opponent_model_frameskip = opponent_model_frameskip

        # Create modules
        self.representation_module = RepresentationModule(
            obs_dim=obs_dim,
            action_dim=action_dim if consider_actions_in_representation else 0,
            opponent_action_dim=opponent_action_dim if consider_actions_in_representation else 0,
            representation_dim=representation_dim,
            hidden_layer_sizes=[int(n) for n in representation_hidden_layer_sizes_specification.split(",")] if representation_hidden_layer_sizes_specification else [],
            hidden_layer_activation=getattr(nn, representation_hidden_layer_activation_specification),
        )
        self.game_model = AbstractGameModel(
            action_dim=action_dim,
            opponent_action_dim=opponent_action_dim,
            obs_dim=representation_dim,
            representation=self.representation_module,
        )
        self.opponent_model = AbstractOpponentModel(
            obs_dim=representation_dim,
            opponent_action_dim=opponent_action_dim,
            representation=self.representation_module,
        )

        self.actor_critic = A2CModule(
            actor=ActorNetwork(
                obs_dim=representation_dim,
                action_dim=action_dim,
                representation=self.representation_module,
            ),
            critic=CriticNetwork(
                obs_dim=representation_dim,
                representation=self.representation_module,
            ),
            optimizer=optimizer,
            **actor_critic_kwargs,
        )

        # To report in the `model` property
        self.full_model = FullModel(
            game_model=self.game_model,
            opponent_model=self.opponent_model,
            actor_critic=self.actor_critic,
        )

        # Optimizers. The actor-critic module already takes care of its own optimizers.
        self.game_model_optimizer = optimizer(self.game_model.parameters(), lr=game_model_learning_rate)
        self.opponent_model_optimizer = optimizer(self.opponent_model.parameters(), lr=opponent_model_learning_rate)

        self.current_observation = None
        self.current_representation = None
        # In case simplified, temporally extended actions are being used. We need to keep track of them
        self.current_simple_action = None
        self.current_simple_action_frame = 0

        # Loss trackers
        self.cumulative_delta = 0
        self.cumulative_delta_n = 0
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
        if len(self.actor_critic.actor.actor_layers) > 1:
            raise ValueError("the actor must be linear to use this method")

        obs_dim = self.representation_dim
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
        
        policy_parameters = dict(self.actor_critic.actor.actor_layers[0].named_parameters())
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
    def act(self, obs: np.ndarray, info: dict) -> int:
        self.current_observation = torch.from_numpy(obs).float().unsqueeze(0)

        # In case a simple action was already being executed, just keep doing it
        if self.current_simple_action is not None:
            self.current_simple_action_frame += 1
            if self.current_simple_action_frame < len(self.current_simple_action):
                action = self.current_simple_action[self.current_simple_action_frame]
                return action
            else:
                self.current_simple_action = None
                self.current_simple_action_frame = 0
        
        action = self.actor_critic.act(self.current_observation)

        # Appropriately convert the action in case it is simplified (not primitive)
        if not self.over_primitive_actions:
            self.current_simple_action = ActionMap.simple_to_discrete(action)
            action = self.current_simple_action[self.current_simple_action_frame]
            
        return action

    def update(self, next_obs: np.ndarray, reward: float, terminated: bool, truncated: bool, info: dict):
        # Setup
        next_obs = torch.from_numpy(next_obs).float().unsqueeze(0)
        #  the actions are from info dictionary at the next step, but that's because it contains which action was performed in the *previous* step
        if not self.over_primitive_actions:
            agent_action = info["p1_move"]
            opponent_action = info["p2_move"]
            agent_action = ActionMap.simple_from_move_index(agent_action)
            opponent_action = ActionMap.simple_from_move_index(opponent_action)
        else:
            raise NotImplementedError("primitive actions are not yet supported")
        agent_action_onehot = torch.nn.functional.one_hot(torch.tensor(agent_action), num_classes=self.action_dim).unsqueeze(0)
        opponent_action_onehot = torch.nn.functional.one_hot(torch.tensor(opponent_action), num_classes=self.opponent_action_dim).unsqueeze(0)

        # Update the different models.
        # The actor and critic are updated first since they already started the training loop in the act() method.
        self._update_actor_critic(self.current_observation, next_obs, reward, terminated)
        self._update_game_model(self.current_observation, agent_action_onehot, opponent_action_onehot, next_obs)
        self._update_opponent_model(self.current_observation, next_obs, opponent_action_onehot)

    # NOTE: the opponent model is also updating the representation function, but it shouldn't
    def _update_opponent_model(self, obs: torch.Tensor, next_obs: torch.Tensor, opponent_action_onehot: torch.Tensor):
        """Calculate the opponent model loss, backpropagate and optimize"""
        if self.opponent_model_learn:
            # NOTE: this assumes learning is being done online, hence why we pick the first row only
            previous_opponent_move_state = FOOTSIES_MOVE_INDEX_TO_MOVE[torch.argmax(obs[0, 17:32]).item()]
            previous_opponent_move_progress = obs[0, 33]
            current_opponent_move_progress = next_obs[0, 33]

            if not self.opponent_model_frameskip or ActionMap.is_state_actionable_late(previous_opponent_move_state, previous_opponent_move_progress, current_opponent_move_progress):
                self.opponent_model_optimizer.zero_grad()

                opponent_action_probabilities = self.opponent_model(obs)
                opponent_model_loss = torch.nn.functional.cross_entropy(opponent_action_probabilities, opponent_action_onehot)
                opponent_model_loss.backward(retain_graph=True)

                self.opponent_model_optimizer.step()

                self.cumulative_loss_opponent_model += opponent_model_loss.item()
                self.cumulative_loss_opponent_model_n += 1

    def _update_game_model(self, obs: torch.Tensor, agent_action_onehot: torch.Tensor, opponent_action_onehot: torch.Tensor, next_obs: torch.Tensor):
        """Calculate the game model loss, backpropagate and optimize"""
        self.game_model_optimizer.zero_grad()

        with torch.no_grad():
            next_representation_target = self.representation_module(next_obs)

        next_representation_predicted = self.game_model(obs, agent_action_onehot, opponent_action_onehot)
        game_model_loss = torch.nn.functional.mse_loss(next_representation_predicted, next_representation_target)
        game_model_loss.backward()

        self.game_model_optimizer.step()

        self.cumulative_loss_game_model += game_model_loss.item()
        self.cumulative_loss_game_model_n += 1

    def _update_actor_critic(self, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool):
        """Calculate the actor-critic loss, backpropagate and optimize"""
        update_actor_critic = self.current_simple_action is None or self.current_simple_action_frame == 0
        if update_actor_critic and self.actor_critic_frameskip:
            # NOTE: this assumes learning is being done online, hence why we pick the first row only
            previous_agent_move_state = FOOTSIES_MOVE_INDEX_TO_MOVE[torch.argmax(self.current_observation[0, 2:17]).item()]
            previous_agent_move_progress = self.current_observation[0, 32]
            current_agent_move_progress = next_obs[0, 32]
            update_actor_critic = ActionMap.is_state_actionable_late(previous_agent_move_state, previous_agent_move_progress, current_agent_move_progress)

        if update_actor_critic:
            self.actor_critic.update(obs, next_obs, reward, terminated)
            self.cumulative_delta += self.actor_critic.delta
            self.cumulative_delta_n += 1

    def load(self, folder_path: str):
        representation_module_path = path.join(folder_path, "representation")
        game_model_path = path.join(folder_path, "game_model")
        actor_critic_path = path.join(folder_path, "actor_critic")
        self.representation_module.load_state_dict(torch.load(representation_module_path))
        self.game_model.load_state_dict(torch.load(game_model_path))
        self.actor_critic.load_state_dict(torch.load(actor_critic_path))

    def save(self, folder_path: str):
        representation_module_path = path.join(folder_path, "representation")
        game_model_path = path.join(folder_path, "game_model")
        actor_critic_path = path.join(folder_path, "actor_critic")
        torch.save(self.representation_module.state_dict(), representation_module_path)
        torch.save(self.game_model.state_dict(), game_model_path)
        torch.save(self.actor_critic.state_dict(), actor_critic_path)

    def extract_policy(self, env: Env) -> Callable[[dict], Tuple[bool, bool, bool]]:
        raise NotImplementedError("policy extraction is not yet supported")

    @property
    def model(self) -> nn.Module:
        return self.full_model

    def evaluate_average_delta(self) -> float:
        res = (
            self.cumulative_delta / self.cumulative_delta_n
        ) if self.cumulative_delta_n != 0 else 0

        self.cumulative_delta = 0
        self.cumulative_delta_n = 0

        return res
    
    def evaluate_average_delta(self) -> float:
        res = (
            self.cumulative_delta / self.cumulative_delta_n
        ) if self.cumulative_delta_n != 0 else 0

        self.cumulative_delta = 0
        self.cumulative_delta_n = 0

        return res
    
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
