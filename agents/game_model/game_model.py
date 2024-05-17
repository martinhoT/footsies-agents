import torch as T
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
        hidden_layer_sizes: list[int] | None = None,
        hidden_layer_activation: type[nn.Module] = nn.LeakyReLU,
        residual: bool = False,
        residual_delta: nn.Module | None = None,
        residual_forget: nn.Module | None = None,
        residual_new: nn.Module | None = None,
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

    def forward(self, obs: T.Tensor, p1_action: T.Tensor, p2_action: T.Tensor) -> T.Tensor:
        if p1_action.dim() < 2:
            p1_action = F.one_hot(p1_action, num_classes=self._p1_action_dim).float()
        if p2_action.dim() < 2:
            p2_action = F.one_hot(p2_action, num_classes=self._p2_action_dim).float()
        
        x = T.hstack((obs, p1_action, p2_action))
        
        if self._residual:    
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

    @property
    def p1_action_dim(self) -> int:
        """Action dimensionality of player 1."""
        return self._p1_action_dim

    @property
    def p2_action_dim(self) -> int:
        """Action dimensionality of player 2."""
        return self._p2_action_dim


class GameModel:
    def __init__(
        self,
        game_model_network: GameModelNetwork,
        learning_rate: float = 1e-2,
        discrete_conversion: bool = False,
        discrete_guard: bool = False,
        by_differences: bool = False,
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
        - `by_differences`: whether to have the game model learn to predict the difference in state rather than the absolute state values
        - `epoch_timesteps`: the number of total timesteps (i.e. update calls) to accumulate before updating
        - `epoch_epochs`: the number of epochs over which to train in a single update
        - `epoch_minibatch_size`: the minibatch size for the accumulated data at each epoch
        """
        self._network = game_model_network
        self._discrete_conversion = discrete_conversion
        self._discrete_guard = discrete_guard
        self._by_differences = by_differences

        self._epoch_timesteps = epoch_timesteps
        self._epoch_epochs = epoch_epochs
        self._epoch_minibatch_size = epoch_minibatch_size

        self.optimizer = T.optim.SGD(params=self._network.parameters(), lr=learning_rate)

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

    def _convert_discrete_components(self, obs: T.Tensor) -> T.Tensor:
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

    def _action_to_tensor(self, action: T.Tensor | int | None, action_dim: int) -> T.Tensor:
        if isinstance(action, int):
            return T.tensor([action])
        
        if action is None:
            return T.zeros(1, action_dim)
    
        return action

    def predict(self, obs: T.Tensor, p1_action: T.Tensor | int | None, p2_action: T.Tensor | int | None) -> T.Tensor:
        """
        Predict the next observation given the current observation and each player's action (either as a probability distribution or as the action ID).
        The result is detached from the computational graph.
        """
        obs = self.preprocess_observation(obs)
        
        p1_action = self._action_to_tensor(p1_action, self._network.p1_action_dim)
        p2_action = self._action_to_tensor(p2_action, self._network.p2_action_dim)

        next_obs = self._network(obs, p1_action, p2_action).detach()
        
        if self._by_differences:
            if not self._discrete_guard:
                next_obs[:, self._guard_p1_slice] = next_obs[:, self._guard_p1_slice] + obs[:, self._guard_p1_slice]
                next_obs[:, self._guard_p2_slice] = next_obs[:, self._guard_p2_slice] + obs[:, self._guard_p2_slice]
            next_obs[:, self._move_progress_slice] = next_obs[:, self._move_progress_slice] + obs[:, self._move_progress_slice]
            next_obs[:, self._position_slice] = next_obs[:, self._position_slice] + obs[:, self._position_slice]

        next_obs = self.postprocess_prediction(next_obs)

        return next_obs

    def preprocess_observation(self, obs: T.Tensor) -> T.Tensor:
        """Preprocess an observation before passing it to the network."""
        return self._convert_discrete_components(obs)

    def postprocess_prediction(self, prediction: T.Tensor) -> T.Tensor:
        """Postprocess a network prediction to fit the expected observation structure."""
        prediction = prediction.clone()
        
        if self._discrete_guard:
            prediction[:, self._guard_p1_slice] = logits_to_probs(prediction[:, self._guard_p1_slice])
            prediction[:, self._guard_p2_slice] = logits_to_probs(prediction[:, self._guard_p2_slice])
        prediction[:, self._move_p1_slice] = logits_to_probs(prediction[:, self._move_p1_slice])
        prediction[:, self._move_p2_slice] = logits_to_probs(prediction[:, self._move_p2_slice])

        # Fix other variables (this should aid in reducing error accumulation when performing multi-step predictions)
        if not self._discrete_guard:
            prediction[:, self._guard_p1_slice].clamp_(0.0, 1.0)
            prediction[:, self._guard_p2_slice].clamp_(0.0, 1.0)
        prediction[:, self._move_progress_slice].clamp_(0.0, 1.0)
        prediction[:, self._position_slice].clamp_(-1.0, 1.0)

        return prediction

    def update(self, obs: T.Tensor, p1_action: T.Tensor | int | None, p2_action: T.Tensor | int | None, next_obs: T.Tensor, *, epoch_data: dict | None = None, actually_update: bool = True) -> tuple[float, float, float, float]:
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
        p1_action = self._action_to_tensor(p1_action, self._network.p1_action_dim)
        p2_action = self._action_to_tensor(p2_action, self._network.p2_action_dim)
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

        if self._by_differences:
            if not self._discrete_guard:
                guard_p1_target = guard_p1_target - obs[:, self._guard_p1_slice]
                guard_p2_target = guard_p2_target - obs[:, self._guard_p2_slice]
            
            move_progress_target = move_progress_target - obs[:, self._move_progress_slice]
            position_target = position_target - obs[:, self._position_slice]

        # Calculate the loss
        if self._discrete_guard:
            guard_p1_loss = F.cross_entropy(predicted[:, self._guard_p1_slice], guard_p1_target)
            guard_p2_loss = F.cross_entropy(predicted[:, self._guard_p2_slice], guard_p2_target)
        else:
            guard_p1_loss = F.mse_loss(predicted[:, self._guard_p1_slice], guard_p1_target)
            guard_p2_loss = F.mse_loss(predicted[:, self._guard_p2_slice], guard_p2_target)

        move_p1_loss = F.cross_entropy(predicted[:, self._move_p1_slice], move_p1_target)
        move_p2_loss = F.cross_entropy(predicted[:, self._move_p2_slice], move_p2_target)
        move_progress_loss = F.mse_loss(predicted[:, self._move_progress_slice], move_progress_target)
        position_loss = F.mse_loss(predicted[:, self._position_slice], position_target)
        
        guard_loss = guard_p1_loss + guard_p2_loss
        move_loss = move_p1_loss + move_p2_loss

        loss = (guard_loss + move_loss + move_progress_loss + position_loss).mean()

        # Backpropagate and step
        if actually_update:
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
        return self._epoch_minibatch_size

    @property
    def learning_rate(self) -> float:
        """The optimizer's learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    @learning_rate.setter
    def learning_rate(self, value: float):
        self.optimizer.param_groups[0]["lr"] = value
