import torch as T
import logging
from torch import nn
from agents.torch_utils import create_layered_network, InputClip, DebugStoreRecent
from collections.abc import Generator
from collections import deque
from dataclasses import astuple, dataclass
from math import log
from contextlib import contextmanager
from typing import Literal

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
        hidden_layer_sizes: list[int] | None = None,
        hidden_layer_activation: type[nn.Module] = nn.LeakyReLU,
        recurrent: nn.RNNBase | bool = False,
    ):
        super().__init__()

        if use_sigmoid_output:
            raise NotImplementedError("sigmoid output is not supported yet")

        self._action_dim = action_dim
        self._use_sigmoid_output = use_sigmoid_output

        if recurrent:
            if isinstance(recurrent, nn.RNNBase):
                self._recurrent = recurrent
            
            else:
                self._recurrent = nn.RNN(
                    input_size=obs_dim,
                    hidden_size=64,
                    num_layers=1,
                    nonlinearity="tanh",
                    bias=True,
                    batch_first=False,
                    dropout=0,
                    bidirectional=False,
                )
            
            layers_input_size = self._recurrent.hidden_size
            
        else:
            self._recurrent = None
            layers_input_size = obs_dim

        self.layers = create_layered_network(
            layers_input_size, action_dim, hidden_layer_sizes, hidden_layer_activation
        )

        self.debug_stores = []

        if input_clip:
            self.layers.append(InputClip(-1, 1, leaky_coef=input_clip_leaky_coef))

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

        # Hidden state if using recurrency
        self._hidden = None

    def _resolve_recurrency(self, obs: T.Tensor, hidden: T.Tensor | None | str = "auto") -> tuple[T.Tensor, T.Tensor | None]:
        if isinstance(hidden, str):
            if hidden == "auto":
                hidden = self.hidden
            else:
                raise ValueError("invalid value for 'hidden', if a string it should be one of {'auto'}")
        
        if self._recurrent is not None:
            x, hidden_state = self._recurrent(obs, hidden)

        else:
            x, hidden_state = obs, None
        
        return x, hidden_state

    def forward(self, obs: T.Tensor, hidden: T.Tensor | None | str = "auto") -> tuple[T.Tensor, T.Tensor | None]:
        x, hidden_state = self._resolve_recurrency(obs, hidden)
        output = self.layers(x)
        return output, hidden_state

    def probabilities(self, obs: T.Tensor, hidden: T.Tensor | None | str = "auto") -> tuple[T.Tensor, T.Tensor]:
        """The action probabilities at the given observation."""
        logits, new_hidden = self(obs, hidden)
        return self.softmax(logits), new_hidden

    def log_probabilities(self, obs: T.Tensor, hidden: T.Tensor | None | str = "auto") -> tuple[T.Tensor, T.Tensor]:
        """The action log-probabilities at the given observation."""
        logits, new_hidden = self(obs, hidden)
        return self.log_softmax(logits), new_hidden

    def distribution(self, obs: T.Tensor, hidden: T.Tensor | None | str = "auto") -> tuple[T.distributions.Categorical, T.Tensor]:
        """The action distribution at the given observation."""
        logits, new_hidden = self(obs, hidden)
        return T.distributions.Categorical(logits=logits), new_hidden

    def compute_hidden_state(self, obs: T.Tensor, hidden: T.Tensor | None | str = "auto") -> T.Tensor | None:
        """Compute only the hidden state, in case one might want to avoid computing probabilities."""
        _, hidden_state = self._resolve_recurrency(obs, hidden)
        return hidden_state

    @property
    def action_dim(self) -> int:
        """The number of actions the network is predicting, i.e. the output dimension."""
        return self._action_dim

    @property
    def hidden(self) -> T.Tensor | None:
        """
        The hidden state of the network, used if recurrency is being used. If `None`, then the hidden state hasn't been initialized (should signify the beginning of an episode).
        
        The internal hidden state can be manually set, but it's discouraged.
        The internal hidden state is already managed by the training loop, to match what the opponent model is being trained on.
        """
        return self._hidden

    @hidden.setter
    def hidden(self, value: T.Tensor | None):
        self._hidden = value

    @property
    def is_recurrent(self) -> bool:
        """Whether the network is recurrent."""
        return self._recurrent is not None

    def update_hidden_state(self, obs: T.Tensor):
        """Update the internal hidden state, useful if using recurrency."""
        hidden = self.compute_hidden_state(obs, self._hidden)
        if hidden is not None:
            # If in the future we need to use the hidden state to compute something, we don't want to backpropagate through this hidden state,
            # or else we would be backpropagating again through the same computational graph and it would get messy.
            # This hidden state is not meant for updating the network, only for inference.
            self._hidden = hidden.detach()

    def reset_hidden_state(self):
        """Reset any hidden state."""
        self._hidden = None


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

        self._obs = T.zeros(max_size, obs_dim)
        self._action = T.zeros(max_size, dtype=T.long)
        self._multiplier = T.zeros(max_size)
        self._loss = T.zeros(max_size)

        self._idx = 0
    
    def _rotate(self):
        """Advance the pointer to the oldest scar."""
        self._idx = (self._idx + 1) % self._max_size
    
    def include(self, obs: T.Tensor, action: int, multiplier: float):
        """Include a new example into the scar store as another example."""
        if self._loss[self._idx] >= self._min_loss:
            LOGGER.warning("The opponent model is forgetting scars! (previous loss: %s, minimum loss: %s). This might be a sign of either a small scar storage or an unforgiving minimum loss.", self._loss[self._idx], self._min_loss)
        
        self._obs[self._idx, :] = obs
        self._action[self._idx] = action
        self._multiplier[self._idx] = multiplier

        self._rotate()

    @property
    def batch(self) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
        """The scar store as a training batch (observations, actions and multipliers)."""
        return self._obs, self._action, self._multiplier

    def update(self, loss: T.Tensor):
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
        scar_store: ScarStore | None = None,
        learning_rate: float = 1e-2,
        loss_dynamic_weights: bool = False,
        loss_dynamic_weights_max: float = 10.0,
        entropy_coef: float = 0.0,
        reset_context_at: Literal["hit", "neutral", "end"] = "end",
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
        - `reset_context_at`: the point at which to reset the opponent model's context/hidden state
        """
        if player_model_network.is_recurrent and scar_store is not None and scar_store.max_size > 1:
            raise ValueError("the scar system should not be used with recurrent networks, as the hidden state is not properly managed")
        if not player_model_network.is_recurrent and scar_store is None:
            raise ValueError("a scar store needs to be specified if using non-recurrent networks")

        self._network = player_model_network
        self._scars = scar_store
        self._loss_dynamic_weights = loss_dynamic_weights
        self._loss_dynamic_weights_max = T.tensor(loss_dynamic_weights_max)
        # In the agent's policy, we perform entropy regularization but not as a linear interpolation.
        # That's because in that case the values in this interpolation have vastly different scales (advantage vs entropy), which makes the coefficient awkward to tune.
        # Here, since we are calculating entropies, we can expect values in similar ranges, so we perform linear interpolation.
        self._entropy_coef = entropy_coef
        self._reset_context_at = reset_context_at
        self._reset_context_at_neutral_dist_threshold = 0.5 # normalized

        self.loss_function = nn.CrossEntropyLoss(reduction="none")

        self.optimizer = T.optim.SGD(params=self._network.parameters(), lr=learning_rate)

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
        self._action_counts = T.zeros(self._network.action_dim)
        self._action_counts_total = 0

        # We need to keep track of the previous observation so that we know whether to reset the model's context in some circumstances.
        # (only relevant if recurrency is being used)
        self._prev_obs = None

        # If the network is recurrent, then we need to keep track of data within an episode before training.
        # We train over an entire episode's worth of data at once.
        self._accumulated_args: list[PlayerModel.AccumulatedUpdate] = []

    def should_reset_context(self, prev_obs: T.Tensor | None, obs: T.Tensor, terminated_or_truncated: bool) -> bool:
        """Whether to reset the context of the opponent model (if recurrent)."""
        # ALWAYS terminate on episode termination, regardless of mode.
        if terminated_or_truncated:
            return True

        # Don't reset hidden state at the beginning.
        if prev_obs is None:
            return False
        
        hit = bool(((obs[0, 0] < prev_obs[0, 0]) or (obs[0, 1] < prev_obs[0, 1])).item())
        
        if self._reset_context_at == "neutral":
            prev_dist = (prev_obs[0, 34] - prev_obs[0, 35]).abs().item()
            dist = (obs[0, 34] - obs[0, 35]).abs().item()
            # We only care about crossing the distance threshold, not staying below it (or else we are constantly resetting the state when under this distance)
            return hit or (dist < self._reset_context_at_neutral_dist_threshold and prev_dist >= self._reset_context_at_neutral_dist_threshold)
        elif self._reset_context_at == "hit":
            return hit
        
        return False

    @dataclass(slots=True)
    class AccumulatedUpdate:
        obs:        T.Tensor
        action:     int
        multiplier: float

    def compute_loss(self, obs: T.Tensor, action: T.Tensor) -> float:
        """
        Compute the loss on the given observation-action pair.
        It is assumed that the observation batch is sequential and context-free when using recurrency.
        """
        predicted, _ = self._network(obs, None)
        distribution = T.distributions.Categorical(logits=predicted)
        loss = (self.loss_function(predicted, action) - self._entropy_coef * distribution.entropy())
        return float(loss.mean().item())

    def update(self, obs: T.Tensor, action: int | None, terminated_or_truncated: bool, multiplier: float = 1.0) -> float | None:
        """
        Update the model to predict the action given the provided observation. Can optionally set a multiplier for the given example to give it more importance.
        If `terminated_or_truncated`, any hidden state related to an episode is reset.

        It's assumed that the observations in `obs` are in sequence, not shuffled in time!

        Returns the loss, or `None` if no learning was performed.
        """
        # If the neural network is recurrent, then we accumulate the arguments over an entire episode until termination
        if self._network.is_recurrent:
            if action is not None:
                # Update the action frequencies
                self._update_action_frequency(action)

                # Update the hidden state with the most recent observation, so that everything else using the network can have up-to-date inference
                # This should match the state sequence that the network is actually trained on! (i.e. with frameskipping)
                self._network.update_hidden_state(obs)

                self._accumulated_args.append(self.AccumulatedUpdate(obs, action, multiplier))
            
            # Don't train as long as the context should be reset
            prev_obs = self._prev_obs
            self._prev_obs = obs
            if not self.should_reset_context(prev_obs, obs, terminated_or_truncated):
                return None

            # Reset the hidden state, it's not even going to be used during training, it's only for access outside of update
            self._network.reset_hidden_state()

            # If we haven't accumulated updates at all (imagine when the player is non-actionable for instance)
            if not self._accumulated_args:
                return None

            obs_batch, action_batch, multiplier_batch = zip(*(astuple(arg) for arg in self._accumulated_args))
            obs_batch: tuple[T.Tensor]
            action_batch: tuple[float]
            multiplier_batch: tuple[float]

            obs = T.vstack(obs_batch).float()
            action_target = T.tensor(action_batch).long()
            multipliers = T.tensor(multiplier_batch).float()

            self._accumulated_args.clear()
        
        else:
            if action is not None:
                # Update the action frequencies
                self._update_action_frequency(action)

                # Update the scar store
                if self._scars is not None:
                    self._scars.include(obs, action, multiplier)
            
            if self._scars is None:
                raise RuntimeError("the scar store should be specified if using non-recurrent networks")
            obs, action_target, multipliers = self._scars.batch

        num_examples = multipliers.nonzero().size(0)

        # Update the network
        self._most_recent_loss = None
        if num_examples > 0:
            self.optimizer.zero_grad()

            predicted, _ = self._network(obs, None)
            distribution = T.distributions.Categorical(logits=predicted)
            loss = (self.loss_function(predicted, action_target) - self._entropy_coef * distribution.entropy()) * multipliers
            # We need to manually perform the mean accoding to how many effective examples we have.
            # Otherwise, the mean will change the speed of learning depending on the scar storage size, which might not be intended.
            loss_agg = loss.sum() / num_examples
            loss_agg.backward()

            self.optimizer.step()

            # Update the scar store with the newest losses
            if not self._network.is_recurrent and self._scars is not None:
                self._scars.update(loss.detach())

            # Check whether learning is dead
            if all(
                not T.any(layer.weight.grad) and not T.any(layer.bias.grad)
                for layer in self.learnable_layers
            ) and loss != 0.0:
                LOGGER.warning("Learning is dead, gradients are 0! (loss: %s)", loss_agg.item())
            
            self._most_recent_loss = float(loss_agg.item())

        # Reset the action counts once an episode has terminated or truncated, as well as the previous observation variable.
        if terminated_or_truncated:
            self._reset_action_counts()
            self._prev_obs = None

        return self._most_recent_loss

    def load(self, path: str):
        self._network.load_state_dict(T.load(path))

    def save(self, path: str):
        T.save(self._network.state_dict(), path)

    def _update_action_frequency(self, action: int):
        """Increase the amount of actions received by the model for updating. Also, update the loss's action weights according to their inverse frequency, if dynamic loss weights are being used."""
        self._action_counts[action] = self._action_counts[action] + 1
        self._action_counts_total += 1

        self._update_loss_function_weights()

    def _reset_action_counts(self):
        """Reset the counts of the actions received by the model for updating. This should be done if the opponent changes, or after some time (if the opponent is adapting)."""
        self._action_counts = T.zeros(self._network.action_dim)
        self._action_counts_total = 0

        self._update_loss_function_weights()

    def _update_loss_function_weights(self):
        """Update the weights of the loss function given to each class (action). No-op if dynamic loss weights are disabled."""
        if self._loss_dynamic_weights:
            # Avoid infinities which are ugly, and too large weights as well.
            self.loss_function.weight = T.min(1 / (self.action_frequencies + 1e-8), self._loss_dynamic_weights_max)

    @property
    def learnable_layers(self) -> Generator[nn.Module, None, None]:
        """The learnable layers of the model."""
        for i, layer in enumerate(self._network.layers):
            if self.is_learnable_layer[i]:
                yield layer

    @property
    def action_frequencies(self) -> T.Tensor:
        """The frequency of each action received by the model for updating."""
        if self._action_counts_total == 0:
            return T.ones_like(self._action_counts) / self._action_counts.size(0)
        return self._action_counts / self._action_counts_total

    @property
    def learning_rate(self) -> float:
        """The optimizer's learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        self.optimizer.param_groups[0]["lr"] = learning_rate

    @property
    def most_recent_loss(self) -> float | None:
        """The loss of the most recent `update` call, or `None` if no update was performed."""
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
    def scars(self) -> ScarStore | None:
        """The scar store of this player model, or `None` if none is being used."""
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
        self._loss_dynamic_weights_max = T.tensor(value)
    
    @property
    def entropy_coef(self) -> float:
        """The entropy regularization coefficient for the loss."""
        return self._entropy_coef
    
    @entropy_coef.setter
    def entropy_coef(self, value: float):
        self._entropy_coef = value
