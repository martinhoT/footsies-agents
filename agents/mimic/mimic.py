import torch
import logging
from torch import nn
from agents.torch_utils import create_layered_network, InputClip, DebugStoreRecent
from collections.abc import Generator
from collections import deque
from dataclasses import dataclass
from math import log

LOGGER = logging.getLogger("main.mimic")


# Sigmoid output is able to tackle action combinations.
# Also, softmax has the problem of not allowing more than one dominant action (for a stochastic agent, for instance), it only focuseds on one of the inputs.
# Softmax also makes the gradients for each output neuron dependent on the values of the other output neurons.
# And, since during training we are not giving the actual probability distributions as targets, it might not be appropriate to train assuming they are.
class PlayerModelNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        use_sigmoid_output: bool = False,
        input_clip: bool = False,
        input_clip_leaky_coef: float = 0,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: type[nn.Module] = nn.LeakyReLU,
    ):
        super().__init__()

        if use_sigmoid_output:
            raise NotImplementedError("sigmoid output is not supported yet")

        self._action_dim = action_dim
        self._use_sigmoid_output = use_sigmoid_output

        self.layers = create_layered_network(
            obs_dim, action_dim, hidden_layer_sizes, hidden_layer_activation
        )

        self.debug_stores = []

        if input_clip:
            self.layers.append(InputClip(-1, 1, leaky_coef=input_clip_leaky_coef))

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.layers(obs)

    def probabilities(self, obs: torch.Tensor) -> torch.Tensor:
        """The action probabilities at the given observation."""
        logits = self(obs)
        return self.softmax(logits)

    def log_probabilities(self, obs: torch.Tensor) -> torch.Tensor:
        """The action log-probabilities at the given observation"""
        logits = self(obs)
        return self.log_softmax(logits)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        """The action distribution at the given observation."""
        return torch.distributions.Categorical(probs=self.probabilities(obs))

    @property
    def action_dim(self) -> int:
        """The number of actions the network is predicting, i.e. the output dimension."""
        return self._action_dim


class ScarStore:
    """
    A storage for scars, i.e. examples that exhibited larger-than-expected loss.

    A normal usage is to, in the online training method:
    - `include` the new example
    - take the `batch` and perform a gradient step on it
    - `update` the scar store
    """

    def __init__(self, obs_dim: int, max_size: int = 1000, min_loss: float = 2.2):
        """
        A storage for scars, i.e. examples that exhibited larger-than-expected loss.

        Parameters
        ----------
        - `max_size`: the maximum number of scars we keep track of
        - `min_loss`: the minimum loss value beyond which we detect a training example as being a scar
        """
        self._max_size = max_size
        self._min_loss = min_loss

        self._obs = torch.zeros(max_size, obs_dim)
        self._action = torch.zeros(max_size, dtype=torch.long)
        self._multiplier = torch.zeros(max_size)
        self._loss = torch.zeros(max_size)

        self._idx = 0
    
    def _rotate(self):
        """Advance the pointer to the oldest scar."""
        self._idx = (self._idx + 1) % self._max_size
    
    def include(self, obs: torch.Tensor, action: int, multiplier: float):
        """Include a new example into the scar store as another example."""
        if self._loss[self._idx] >= self._min_loss:
            LOGGER.warning("The opponent model is forgetting scars! (previous loss: %s, minimum loss: %s). This might be a sign of either a small scar storage or an unforgiving minimum loss.", self._loss[self._idx], self._min_loss)
        
        self._obs[self._idx, :] = obs
        self._action[self._idx] = action
        self._multiplier[self._idx] = multiplier

        self._rotate()

    @property
    def batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The scar store as a training batch (observations, actions and multipliers)."""
        return self._obs, self._action, self._multiplier

    def update(self, loss: torch.Tensor):
        """Update the scar storage. The `loss` should be unreduced (e.g. no averaging over the batch), and thus have the same batch size as the other arguments."""
        self._loss = loss
        # Invalidate examples that already have acceptable loss.
        # Setting the multiplier to 0 will make the example innefectual during training, and will effectively be ignored.
        self._multiplier[self._loss < self._min_loss] = 0.0
    
    @property
    def min_loss(self) -> float:
        """The minimum loss required to classify a training example as being a scar."""
        return self._min_loss
    
    @min_loss.setter
    def min_loss(self, min_loss: float):
        self._min_loss = min_loss

    @property
    def max_size(self) -> int:
        """The maximum number of scars we keep track of."""
        return self._max_size


class PlayerModel:
    def __init__(
        self,
        player_model_network: PlayerModelNetwork,
        scar_store: ScarStore = None,
        learning_rate: float = 1e-2,
        loss_dynamic_weights: bool = False,
        loss_dynamic_weights_max: float = 10.0,
        entropy_coef: float = 0.0,
    ):
        """
        Player model for predicting actions given an observation.
        The model is trained using supervised learning.

        This model implements a "scar" system.
        A scar is a training example that had a loss spike, which we should keep in mind and keep training on in the future.
        Contrary to the previous idea of reinforcement, scarring just executes one training step per update, but does so in a batch including scars.
        This should also come as a replacement for custom move importances.
        Scars are constantly trained with the same batch size.
        Although this is wasteful, it may also be more efficient since the operations are vectorized.
        It should also provide consistent performance.
        
        Parameters
        ----------
        - `player_model_network`: the predictor network
        - `scar_store`: the scar store. If `None`, scars won't be used
        - `learning_rate`: the optimizer's learning rate
        - `loss_dynamic_weights`: whether to update the loss weights dynamically based on the action frequencies
        - `loss_dynamic_weights_max`: cap the loss weights to a maximum value, if dynamic weights are used.
        Since some actions can be very infrequent, it's easy for its weights to increase too much, creating gradients that are too big.
        This parameter alleviates this problem
        - `entropy_coef`: the entropy regularization coefficient for the loss, implemented as a linear interpolation between the cross-entropy and the predicted distribution's entropy.
        If 0, no entropy regularization is added to the loss.
        If 1, then only entropy is maximized
        """
        self._network = player_model_network
        self._scars = scar_store
        self._loss_dynamic_weights = loss_dynamic_weights
        self._loss_dynamic_weights_max = torch.tensor(loss_dynamic_weights_max)
        # In the agent's policy, we perform entropy regularization but not as a linear interpolation.
        # That's because in that case the values in this interpolation have vastly different scales (advantage vs entropy), which makes the coefficient awkward to tune.
        # Here, since we are calculating entropies, we can expect values in similar ranges.
        self._entropy_coef = entropy_coef

        self.loss_function = nn.CrossEntropyLoss(reduction="none")

        self.optimizer = torch.optim.SGD(params=self._network.parameters(), lr=learning_rate)

        # Just to make training easier, know which layers actually have learnable parameters
        self.is_learnable_layer = [
            ("weight" in param_names and "bias" in param_names)
            for param_names in map(
                lambda params: map(lambda t: t[0], params),
                map(
                    list,
                    map(
                        nn.Module.named_parameters,
                        self._network.layers
                    ),
                )
            )
        ]

        # For tracking purposes
        self._most_recent_loss = 0.0

        # Action counts
        self._action_counts = torch.zeros(self._network.action_dim)
        self._action_counts_total = 0

    def update(self, obs: torch.Tensor, action: torch.Tensor | int, multiplier: torch.Tensor | float = 1.0) -> float:
        """
        Update the model to predict the action given the provided observation. Can optionally set a multiplier for the given example to give it more importance.

        Returns the loss.
        """
        # Update the action frequencies
        self._update_action_frequency(action)

        # Update the scar store
        self._scars.include(obs, action, multiplier)
        obs, action, multiplier = self._scars.batch

        # Update the network
        self.optimizer.zero_grad()

        predicted = self._network(obs)
        distribution = torch.distributions.Categorical(logits=predicted)
        loss = (self.loss_function(predicted, action) * (1 - self._entropy_coef) + -distribution.entropy() * self._entropy_coef) * multiplier
        # We need to manually perform the mean accoding to how many effective examples we have.
        # Otherwise, the mean will change the speed of learning depending on the scar storage size, which might not be intended
        loss_agg = loss.sum() / multiplier.nonzero().size(dim=0)
        loss_agg.backward()

        self.optimizer.step()

        # Update the scar store with the newest losses
        self._scars.update(loss)

        # Check whether learning is dead
        if all(
            not torch.any(layer.weight.grad) and not torch.any(layer.bias.grad)
            for layer in self.learnable_layers
        ):
            LOGGER.warning("Learning is dead, gradients are 0! (loss: %s)", loss_agg.item())

        self._most_recent_loss = loss_agg.item()
        return self._most_recent_loss

    def predict(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Sample an action from the predicted action distribution at the given observation. If deterministic, the action with the highest probability is chosen."""
        with torch.no_grad():
            probs = self.probabilities(obs)

            if deterministic:
                return torch.argmax(probs, axis=-1)
            else:
                distribution = torch.distributions.Categorical(probs=probs)
                return distribution.sample()

    def probabilities(self, obs: torch.Tensor) -> torch.Tensor:
        """Get the predicted action probabilities at the given observation. The probabilities are detached from any computation graph."""
        return self._network.probabilities(obs).detach()

    def load(self, path: str):
        self._network.load_state_dict(torch.load(path))

    def save(self, path: str):
        torch.save(self._network.state_dict(), path)

    def _update_action_frequency(self, action: int):
        """Increase the amount of actions received by the model for updating. Also, update the loss's action weights according to their inverse frequency, if dynamic loss weights are being used."""
        self._action_counts[action] = self._action_counts[action] + 1
        self._action_counts_total += 1

        if self._loss_dynamic_weights:
            # Avoid infinities which are ugly, and too large weights as well
            self.loss_function.weight = torch.min(1 / (self.action_frequencies + 1e-8), self._loss_dynamic_weights_max)
        
        else:
            self.loss_function.weight = None

    def _reset_action_counts(self):
        """Reset the counts of the actions received by the model for updating. This should be done if the opponent changes, or after some time (if the opponent is adapting)."""
        self._action_counts = torch.zeros(self._network.action_dim)
        self._action_counts_total = 0

        if self._loss_dynamic_weights:
            # Avoid infinities which are ugly, and too large weights as well
            self.loss_function.weight = torch.min(1 / (self.action_frequencies + 1e-8), self._loss_dynamic_weights_max)
        
        else:
            self.loss_function.weight = None

    @property
    def learnable_layers(self) -> Generator[nn.Module, None, None]:
        """The learnable layers of the model."""
        for i, layer in enumerate(self._network.layers):
            if self.is_learnable_layer[i]:
                yield layer

    @property
    def action_frequencies(self) -> torch.Tensor:
        """The frequency of each action received by the model for updating."""
        if self._action_counts_total == 0:
            return torch.ones_like(self._action_counts) / self._action_counts.size(0)
        return self._action_counts / self._action_counts_total

    @property
    def learning_rate(self) -> float:
        """The optimizer's learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        self.optimizer.param_groups[0]["lr"] = learning_rate

    @property
    def most_recent_loss(self) -> float:
        """The loss of the most recent `update` call."""
        return self._most_recent_loss
    
    @property
    def number_of_scars(self) -> int:
        """The total number of scars in effect."""
        return self._scars._multiplier.nonzero().size(0) if self._scars is not None else 0

    @property
    def network(self) -> PlayerModelNetwork:
        """The predictor network of this player model."""
        return self._network
    
    @property
    def scars(self) -> ScarStore:
        """The scar store of this player model."""
        return self._scars

    @property
    def loss_dynamic_weights(self) -> bool:
        """Whether the loss weights are updated dynamically based on the action frequencies."""
        return self._loss_dynamic_weights

    @loss_dynamic_weights.setter
    def loss_dynamic_weights(self, value: bool):
        self._loss_dynamic_weights = value

    @property
    def loss_dynamic_weights_max(self) -> float:
        """The maximum weight attributed when using dynamic class weights for the loss."""
        return self._loss_dynamic_weights_max.item()

    @loss_dynamic_weights_max.setter
    def loss_dynamic_weights_max(self, value: float):
        self._loss_dynamic_weights_max = torch.tensor(value)
    
    @property
    def entropy_coef(self) -> float:
        """The entropy regularization coefficient for the loss."""
        return self._entropy_coef
    
    @entropy_coef.setter
    def entropy_coef(self, value: float):
        self._entropy_coef = value