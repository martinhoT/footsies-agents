import numpy as np
from torch import nn
from agents.action import ActionMap
from agents.base import FootsiesAgentBase
from gymnasium import Env, Space
from typing import Callable, Tuple
from agents.the_one.model import RepresentationModule, AbstractGameModel, AbstractOpponentModel
from agents.a2c.a2c import A2CModule, ActorNetwork, CriticNetwork


class FootsiesAgent(FootsiesAgentBase):
    def __init__(
        self,
        observation_space_size: Space,
        action_space_size: Space,
        representation_dim: int,
        representation_hidden_layer_sizes_specification: str = "",
        representation_hidden_layer_activation_specification: nn.Module = nn.Identity,
        over_primitive_actions: bool = False,
    ):
        obs_dim = observation_space_size
        action_dim = action_space_size if over_primitive_actions else ActionMap.n_simple()
        opponent_action_dim = action_space_size if over_primitive_actions else ActionMap.n_simple()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.opponent_action_dim = opponent_action_dim
        self.representation_dim = representation_dim

        self.representation_module = RepresentationModule(
            obs_dim=obs_dim,
            action_dim=action_dim,
            opponent_action_dim=opponent_action_dim,
            representation_dim=representation_dim,
            hidden_layer_sizes=[int(n) for n in representation_hidden_layer_sizes_specification.split(",")] if representation_hidden_layer_sizes_specification else [],
            hidden_layer_activation=getattr(nn, representation_hidden_layer_activation_specification),
        )
        self.policy_value = A2CModule(
            actor=ActorNetwork(
                obs_dim=representation_dim,
                action_dim=action_dim
            )
        )
        self.game_model = AbstractGameModel(
            representation_dim=representation_dim
        )
        self.opponent_model = AbstractOpponentModel(representation_dim, opponent_action_dim)

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
        
        policy_parameters = dict.fromkeys(self.policy_value.actor.named_parameters())
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

    
    def act(self, obs) -> "any":
        ...

    def update(self, next_obs, reward: float, terminated: bool, truncated: bool, info: dict):
        ...

    # def preprocess(self, env: Env):
    #     ...

    def load(self, folder_path: str):
        ...

    def save(self, folder_path: str):
        ...

    def extract_policy(self, env: Env) -> Callable[[dict], Tuple[bool, bool, bool]]:
        ...