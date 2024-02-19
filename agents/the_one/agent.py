import numpy as np
import torch
from torch import nn
from os import path
from agents.action import ActionMap
from agents.base import FootsiesAgentBase
from gymnasium import Env
from typing import Callable, Tuple
from agents.the_one.model import RepresentationModule, AbstractGameModel, AbstractOpponentModel
from agents.the_one.reaction_time import ReactionTimeEmulator
from agents.a2c.a2c import A2CModule, ActorNetwork, CriticNetwork
from footsies_gym.moves import FOOTSIES_MOVE_INDEX_TO_MOVE


class FootsiesAgent(FootsiesAgentBase):
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
        representation_learning_rate: float = 1e-4,
        game_model_learning_rate: float = 1e-4,
        opponent_model_learning_rate: float = 1e-4,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 1e-4,
        # Other modules
        reaction_time_emulator: ReactionTimeEmulator = None,
    ):
        if over_primitive_actions:
            raise NotImplementedError("primitive actions are not yet supported")
        
        if reaction_time_emulator is None:
            reaction_time_emulator = ReactionTimeEmulator(
                inaction_probability=0.0,
                multiplier=1.0,
                additive=0.0,
                history_size=30,
            )

        obs_dim = observation_space_size
        action_dim = action_space_size if over_primitive_actions else ActionMap.n_simple()
        opponent_action_dim = action_space_size if over_primitive_actions else ActionMap.n_simple()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.opponent_action_dim = opponent_action_dim
        self.representation_dim = representation_dim
        self.game_model_double_grad = game_model_double_grad
        self.over_primitive_actions = over_primitive_actions
        self.opponent_model_learn = opponent_model_learn
        self.actor_critic_frameskip = actor_critic_frameskip
        self.opponent_model_frameskip = opponent_model_frameskip

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
            representation_dim=representation_dim,
        )
        self.opponent_model = AbstractOpponentModel(
            representation_dim=representation_dim,
            opponent_action_dim=opponent_action_dim,
        )

        self.actor_critic = A2CModule(
            actor=ActorNetwork(
                obs_dim=representation_dim,
                action_dim=action_dim
            ),
            critic=CriticNetwork(
                obs_dim=representation_dim,
            ),
            discount=1.0,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_eligibility_traces_decay=0.0,
            critic_eligibility_traces_decay=0.0,
            actor_entropy_loss_coef=0.0,
            optimizer=optimizer,
        )

        # Optimizers. The actor-critic module already takes care of its own optimizers.
        self.representation_optimizer = optimizer(self.representation_module.parameters(), lr=representation_learning_rate)
        self.game_model_optimizer = optimizer(self.representation_module.parameters(), lr=game_model_learning_rate)
        self.opponent_model_optimizer = optimizer(self.representation_module.parameters(), lr=opponent_model_learning_rate)

        self.current_observation = None
        self.current_representation = None
        # In case simplified, temporally extended actions are being used. We need to keep track of them
        self.current_simple_action = None

    def env_concat(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Obtain the concatenated weights that calculate the next environment observation `n` steps into the future.
        Only works if the game model, opponent model and agent policy are linear.

        Parameters
        ----------
        - `n`: the number of steps to predict

        Returns
        -------
        - `s_mtx`: the matrix to multiply with the observation `s`
        - `a_mtx`: the matrix to multiply with the agent action `a`
        - `o_mtx`: the matrix to multiply with the opponent action `o`
        - `bias`: the bias

        The final step is computed as `s_mtx @ s + a_mtx @ a + o_mtx @ o + bias`
        """
        # Matrices for implementing the concatenation of input vectors
        C_s = np.zeros((self.obs_dim + self.action_dim + self.opponent_action_dim, self.obs_dim))
        C_a = np.zeros((self.obs_dim + self.action_dim + self.opponent_action_dim, self.action_dim))
        C_o = np.zeros((self.obs_dim + self.action_dim + self.opponent_action_dim, self.opponent_action_dim))

        C_s[:self.obs_dim, :] = np.eye(self.obs_dim)
        C_a[self.obs_dim:self.action_dim, :] = np.eye(self.action_dim)
        C_o[:self.obs_dim + self.action_dim:self.opponent_action_dim, :] = np.eye(self.opponent_action_dim)

        # Game model, opponent model and agent policy parameters
        game_model_parameters = dict.fromkeys(self.game_model.named_parameters())
        W_g = game_model_parameters["weight"]
        b_g = game_model_parameters["bias"]
        
        opponent_model_parameters = dict.fromkeys(self.opponent_model.named_parameters())
        W_o = opponent_model_parameters["weight"]
        b_o = opponent_model_parameters["bias"]
        
        policy_parameters = dict.fromkeys(self.actor_critic.actor.named_parameters())
        W_a = policy_parameters["weight"]
        b_a = policy_parameters["bias"]

        # NOTE: may be wrong, haven't thoroughly checked because I'm tired
        """
        s_n = ((W_g @ C_s) ^ n + (W_g @ C_a) ^ (n - 1) @ W_a + (W_g @ C_o) ^ (n - 1) @ W_o) @ s_0 
            + (W_g @ C_s) ^ (n - 1) @ W_g @ C_a @ a_0
            + (W_g @ C_s) ^ (n - 1) @ W_g @ C_o @ o_0
            + (W_g @ C_s) ^ (n - 1) @ b_g
            + (W_g @ C_a) ^ (n - 1) @ b_a
            + (W_g @ C_o) ^ (n - 1) @ b_o
            + b_g
        """ 
        
        s_mtx = (
            + np.linalg.matrix_power(W_g @ C_s, n)
            + np.linalg.matrix_power(W_g @ C_a, n - 1) @ W_a
            + np.linalg.matrix_power(W_g @ C_o, n - 1) @ W_o
        )

        a_mtx = np.linalg.matrix_power(W_g @ C_s, n - 1) @ W_g @ C_a

        o_mtx = np.linalg.matrix_power(W_g @ C_s, n - 1) @ W_g @ C_o

        bias = (
            + np.linalg.matrix_power(W_g @ C_s, n - 1) @ b_g
            + np.linalg.matrix_power(W_g @ C_a, n - 1) @ b_a
            + np.linalg.matrix_power(W_g @ C_o, n - 1) @ b_o
            + b_g
        )

        return s_mtx, a_mtx, o_mtx, bias

    # It's in this function that the current observation and representation variables are updated
    def act(self, obs: np.ndarray, info: dict) -> int:
        obs = torch.from_numpy(obs)
        self.current_observation = obs
        representation = self.representation_module(obs)
        self.current_representation = representation

        # In case a simple action was already being executed, just keep doing it
        if not self.over_primitive_actions:
            try:
                action = next(self.current_simple_action)
                return action
            except StopIteration:
                self.current_simple_action = None
        
        action = self.actor_critic.act(representation)

        # Appropriately convert the action in case it is simplified (not primitive)
        if not self.over_primitive_actions:
            self.current_simple_action = iter(ActionMap.simple_to_discrete())
            action = next(self.current_simple_action)
            
        return action

    def update(self, next_obs: np.ndarray, reward: float, terminated: bool, truncated: bool, info: dict):
        # Setup
        next_obs = torch.from_numpy(next_obs)
        #  the actions are from info dictionary at the next step, but that's because it contains which action was performed in the *previous* step
        if not self.over_primitive_actions:
            agent_action = info["p1_move"]
            opponent_action = info["p2_move"]
            agent_action = ActionMap.simple_from_move_index(agent_action)
            opponent_action = ActionMap.simple_from_move_index(opponent_action)
        else:
            raise NotImplementedError("primitive actions are not yet supported")
        agent_action_onehot = torch.nn.functional.one_hot(torch.tensor(agent_action), num_classes=self.action_dim)
        opponent_action_onehot = torch.nn.functional.one_hot(torch.tensor(opponent_action), num_classes=self.opponent_action_dim)

        # Zero the gradients
        self.representation_optimizer.zero_grad()
        self.game_model_optimizer.zero_grad()
        if self.opponent_model_learn:
            self.opponent_model_optimizer.zero_grad()

        # Get the shared representation
        next_representation = self.representation_module(next_obs)

        # Calculate the opponent model loss and backpropagate
        opponent_model_backpropagated = False
        if self.opponent_model_learn:
            # NOTE: this assumes learning is being done online, hence why we pick the first row only
            previous_opponent_move_state = FOOTSIES_MOVE_INDEX_TO_MOVE[torch.argmax(self.current_observation[0, 17:32]).item()]
            previous_opponent_move_progress = self.current_observation[0, 33]
            current_opponent_move_progress = next_obs[0, 33]

            if not self.opponent_model_frameskip or ActionMap.is_state_actionable_late(previous_opponent_move_state, previous_opponent_move_progress, current_opponent_move_progress):
                opponent_action_probabilities = self.opponent_model(self.current_representation)
                opponent_model_loss = torch.nn.functional.cross_entropy(opponent_action_probabilities, opponent_action_onehot)
                opponent_model_loss.backward(retain_graph=True)
                opponent_model_backpropagated = True

        # Calculate the game model loss and backpropagate
        game_model_input = torch.hstack((self.current_representation, agent_action_onehot, opponent_action_onehot))
        next_representation_predicted = self.game_model(game_model_input)
        next_representation_target = next_representation
        if self.game_model_double_grad:
            next_representation_target = next_representation_target.detach()
        game_model_loss = torch.nn.functional.mse_loss(next_representation_predicted, next_representation_target)
        game_model_loss.backward(retain_graph=True)

        # Calculate the actor-critic loss, backpropagate and optimize
        update_actor_critic = True
        if self.actor_critic_frameskip:
            # NOTE: this assumes learning is being done online, hence why we pick the first row only
            previous_agent_move_state = FOOTSIES_MOVE_INDEX_TO_MOVE[torch.argmax(self.current_observation[0, 2:17]).item()]
            previous_agent_move_progress = self.current_observation[0, 32]
            current_agent_move_progress = next_obs[0, 32]
            update_actor_critic = ActionMap.is_state_actionable_late(previous_agent_move_state, previous_agent_move_progress, current_agent_move_progress)

        if update_actor_critic:
            if self.current_simple_action is not None:
                raise RuntimeError("the simple action that was executed should have already finished before an actor-critic update, frame skipping is ill-formed")
            self.actor_critic.update(self.current_representation, next_representation, reward, terminated)

        # Optimize everything else
        if self.opponent_model_learn and opponent_model_backpropagated:
            self.opponent_model_optimizer.step()
        self.game_model_optimizer.step()
        self.representation_module.step()

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
        ...