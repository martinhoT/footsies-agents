import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from agents.torch_utils import create_layered_network
from agents.torch_utils import epoched
from torch.distributions.utils import probs_to_logits, logits_to_probs


class GameModelNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        p1_action_dim: int,
        p2_action_dim: int,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: type[nn.Module] = nn.LeakyReLU,
        residual: bool = False,
        residual_delta: nn.Module = None,
        residual_forget: nn.Module = None,
        residual_new: nn.Module = None,
    ):
        """
        Neural network for predicting the next FOOTSIES observation given the current observation and the action of each player.

        Parameters
        ----------
        - `obs_dim`: the dimension of the observation space
        - `p1_action_dim`: the dimension of the agent's action space    
        - `p2_action_dim`: the dimension of the opponent's action space
        - `hidden_layer_sizes`: list of hidden layer sizes
        - `hidden_layer_activation`: activation function for the hidden layers
        - `residual`: whether to use a residual-style architecture in the SSBM paper "At Human Speed: Deep Reinforcement Learning with Action Delay"
        - `residual_delta`: the delta network. If `None`, the architecture specified in the `hidden_layer_sizes` and `hidden_layer_activation` arguments will be used
        - `residual_forget`: the forget network. If `None`, the architecture specified in the `hidden_layer_sizes` and `hidden_layer_activation` arguments will be used
        - `residual_new`: the new network. If `None`, the architecture specified in the `hidden_layer_sizes` and `hidden_layer_activation` arguments will be used
        """
        super().__init__()
        
        input_dim = obs_dim + p1_action_dim + p2_action_dim
        
        self._residual = residual
        self._p1_action_dim = p1_action_dim
        self._p2_action_dim = p2_action_dim

        if residual:
            self._delta_network = residual_delta if residual_delta is not None else create_layered_network(
                input_dim, obs_dim, hidden_layer_sizes, hidden_layer_activation
            )
            self._forget_network = residual_forget if residual_forget is not None else create_layered_network(
                input_dim, obs_dim, hidden_layer_sizes, hidden_layer_activation
            )
            self._new_network = residual_new if residual_new is not None else create_layered_network(
                input_dim, obs_dim, hidden_layer_sizes, hidden_layer_activation
            )

        else:
            self._layers = create_layered_network(
                input_dim, obs_dim, hidden_layer_sizes, hidden_layer_activation
            )

    def forward(self, obs: torch.Tensor, p1_action: torch.Tensor, p2_action: torch.Tensor) -> torch.Tensor:
        if p1_action.dim() < 2:
            p1_action = F.one_hot(p1_action, num_classes=self._p1_action_dim).float()
        if p2_action.dim() < 2:
            p2_action = F.one_hot(p2_action, num_classes=self._p2_action_dim).float()
        
        if self._residual:
            x = torch.hstack((obs, p1_action, p2_action))
            
            f = self._forget_network(x)
            d = self._delta_network(x)
            n = self._new_network(x)

            res = f * (obs + d) + (1 - f) * n

        else:
            res = self._layers(x)
        
        return res

    @property
    def residual(self) -> bool:
        """Whether the network uses a residual architecture."""
        return self._residual


class GameModel:
    def __init__(
        self,
        game_model_network: GameModelNetwork,
        learning_rate: float = 1e-2,
        discrete_conversion: bool = False,
        discrete_guard: bool = False,
        epoch_timesteps: int = 1,
        epoch_epochs: int = 1,
        epoch_minibatch_size: int = 1,
    ):
        """
        Game model agent for modeling the deterministic dynamics of the FOOTSIES environment.

        The input of the model is the environment observation, agent action and opponent action.
        The actions are simplified by default.
        The output of the model is a prediction for the next environment observation.

        Parameters
        ----------
        - `discrete_conversion`: whether to convert the discrete components of the observation (i.e. the moves) to logits before feeding them to the network
        - `discrete_guard`: whether the guard variable is discrete, and should be treated as such
        - `epoch_timesteps`: the number of total timesteps (i.e. update calls) to accumulate before updating
        - `epoch_epochs`: the number of epochs over which to train in a single update
        - `epoch_minibatch_size`: the minibatch size for the accumulated data at each epoch
        """
        self._network = game_model_network
        self._discrete_conversion = discrete_conversion
        self._discrete_guard = discrete_guard

        self._epoch_timesteps = epoch_timesteps
        self._epoch_epochs = epoch_epochs
        self._epoch_minibatch_size = epoch_minibatch_size

        self.optimizer = torch.optim.SGD(params=self._network.parameters(), lr=learning_rate)

        # Slices at which we get the desired variables of the input.
        # This is necessary in case, for instance, we consider the guard variable to be discrete rather than continuous.
        if discrete_guard:
            self._guard_p1_slice = slice(0, 4)
            self._guard_p2_slice = slice(4, 8)
            self._move_p1_slice = slice(8, 23)
            self._move_p2_slice = slice(23, 38)
            self._move_progress_slice = slice(38, 40)
            self._position_slice = slice(40, 42)

        else:
            self._guard_p1_slice = slice(0, 1)
            self._guard_p2_slice = slice(1, 2)
            self._move_p1_slice = slice(2, 17)
            self._move_p2_slice = slice(17, 32)
            self._move_progress_slice = slice(32, 34)
            self._position_slice = slice(34, 36)

        # For training
        self.state_batch_as_list = []
        self.current_observation = None

    def _convert_discrete_components(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Convert the discrete components of the observation to logits.
        This makes it so that addition is more meaningful especially when applying the residual architecture, as suggested in the SSBM paper.
        
        This function is applied manually during training, both to the input observation and the target observation.
        """
        if self._discrete_conversion:
            obs = obs.clone()
            if self._discrete_guard:
                obs[:, self._move_p1_slice] = probs_to_logits(obs[:, self._move_p1_slice])
                obs[:, self._move_p2_slice] = probs_to_logits(obs[:, self._move_p2_slice])
            obs[:, self._move_p1_slice] = probs_to_logits(obs[:, self._move_p1_slice])
            obs[:, self._move_p2_slice] = probs_to_logits(obs[:, self._move_p2_slice])
        
        return obs

    def predict(self, obs: torch.Tensor, p1_action: torch.Tensor | int, p2_action: torch.Tensor | int, raw: bool = False) -> torch.Tensor:
        """Predict the next observation given the current observation and each player's action (either as a probability distribution or as the action ID). The result is detached from the computational graph."""
        obs = self.preprocess_observation(obs)
        
        next_obs = self._network(obs, p1_action, p2_action).detach()
        
        next_obs = self.postprocess_prediction(next_obs)

        return next_obs

    def preprocess_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Preprocess an observation before passing it to the network."""
        return self._convert_discrete_components(obs)

    def postprocess_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        """Postprocess a network prediction to fit the expected observation structure."""
        prediction = prediction.clone()
        
        if self._discrete_guard:
            prediction[:, self._guard_p1_slice] = logits_to_probs(prediction[:, self._guard_p1_slice])
            prediction[:, self._guard_p2_slice] = logits_to_probs(prediction[:, self._guard_p2_slice])
        prediction[:, self._move_p1_slice] = logits_to_probs(prediction[:, self._move_p1_slice])
        prediction[:, self._move_p2_slice] = logits_to_probs(prediction[:, self._move_p2_slice])

        return prediction

    def update(self, obs: torch.Tensor, p1_action: torch.Tensor | int, p2_action: torch.Tensor | int, next_obs: torch.Tensor, *, epoch_data: dict | None = None) -> tuple[float, float, float, float]:
        """
        Update the game model with the given transition.
        
        Returns
        -------
        - `guard_loss`: the loss for the guard variable
        - `move_loss`: the loss for move prediction of both players
        - `move_progress_loss`: the loss for the move progress variable
        - `position_loss`: the loss for the position variable
        """
        # Pre-process the input and target
        obs = self._convert_discrete_components(obs)
        next_obs = self._convert_discrete_components(next_obs)

        # Obtain prediction
        if isinstance(p1_action, int):
            p1_action = torch.tensor([p1_action])
        if isinstance(p2_action, int):
            p2_action = torch.tensor([p2_action])
        predicted = self._network(obs, p1_action, p2_action)
        
        # Prepare the targets
        guard_p1_target = next_obs[:, self._guard_p1_slice]
        guard_p2_target = next_obs[:, self._guard_p2_slice]
        if self._discrete_guard:
            guard_p1_target = guard_p1_target.argmax(dim=-1)
            guard_p2_target = guard_p2_target.argmax(dim=-1)

        move_p1_target = next_obs[:, self._move_p1_slice].argmax(dim=-1)
        move_p2_target = next_obs[:, self._move_p2_slice].argmax(dim=-1)

        move_progress_target = next_obs[:, self._move_progress_slice]
        position_target = next_obs[:, self._position_slice]

        # Calculate the loss
        if self._discrete_guard:
            guard_p1_loss = F.cross_entropy(predicted[:, self._guard_p1_slice], guard_p1_target)
            guard_p2_loss = F.cross_entropy(predicted[:, self._guard_p2_slice], guard_p2_target)
        else:
            guard_p1_loss = F.mse_loss(predicted[:, self._guard_p1_slice], guard_p1_target)
            guard_p2_loss = F.mse_loss(predicted[:, self._guard_p2_slice], guard_p2_target)

        move_p1_loss = F.cross_entropy(predicted[:, self._move_p1_slice], move_p1_target)
        move_p2_loss = F.cross_entropy(predicted[:, self._move_p1_slice], move_p2_target)
        move_progress_loss = F.mse_loss(predicted[:, self._move_progress_slice], move_progress_target)
        position_loss = F.mse_loss(predicted[:, self._position_slice], position_target)
        
        guard_loss = guard_p1_loss + guard_p2_loss
        move_loss = move_p1_loss + move_p2_loss

        loss = (guard_loss + move_loss + move_progress_loss + position_loss).mean()

        # Backpropagate and step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return guard_loss.mean().item(), move_loss.mean().item(), move_progress_loss.mean().item(), position_loss.mean().item()

    @property
    def network(self) -> GameModelNetwork:
        """The inner neural network."""
        return self._network
    
    @property
    def epoch_timesteps(self) -> int:
        """The number of timesteps (i.e. function calls) to accumulate before training."""
        return self._epoch_timesteps
    
    @property
    def epoch_epochs(self) -> int:
        """The number of epochs to train on the accumulated data."""
        return self._epoch_epochs

    @property
    def epoch_minibatch_size(self) -> int:
        """The size of the accumulated data partitions."""
        return self._minibatch_size

    @property
    def learning_rate(self) -> float:
        """The optimizer's learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    @learning_rate.setter
    def learning_rate(self, value: float):
        self.optimizer.param_groups[0]["lr"] = value
