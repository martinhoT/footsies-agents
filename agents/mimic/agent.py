from copy import deepcopy
import os
import numpy as np
import torch
from torch import nn
from agents.base import FootsiesAgentBase
from agents.torch_utils import create_layered_network, InputClip, DebugStoreRecent
from agents.action import ActionMap
from gymnasium import Env
from typing import Callable, List, Tuple
from gymnasium import Space
from footsies_gym.moves import FootsiesMove, FOOTSIES_MOVE_INDEX_TO_MOVE


# Sigmoid output is able to tackle action combinations.
# Also, softmax has the problem of not allowing more than one dominant action (for a stochastic agent, for instance), it only focuses on one of the inputs.
# Softmax also makes the gradients for each output neuron dependent on the values of the other output neurons.
# And, since during training we are not giving the actual probability distributions as targets, it might not be appropriate to train assuming they are.
class PlayerModelNetwork(nn.Module):
    DEBUG = False

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_sigmoid_output: bool = False,
        input_clip: bool = False,
        input_clip_leaky_coef: float = 0,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: nn.Module = nn.LeakyReLU,
    ):
        super().__init__()

        self.layers = create_layered_network(
            input_dim, output_dim, hidden_layer_sizes, hidden_layer_activation
        )

        self.debug_stores = []

        if input_clip:
            self.layers.append(InputClip(leaky_coef=input_clip_leaky_coef))
            if self.DEBUG:
                debug_store = DebugStoreRecent()
                self.debug_stores.append(debug_store)
                self.layers.append(debug_store)

        self.layers.append(
            nn.Sigmoid() if use_sigmoid_output else nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class PlayerModel:
    def __init__(
        self,
        obs_size: int,
        n_moves: int,
        obs_mask: torch.Tensor = None,
        mini_batch_size: int = 1000,
        use_sigmoid_output: bool = False,
        input_clip: bool = False,
        input_clip_leaky_coef: float = 0,
        hidden_layer_sizes: list[int] = None,
        hidden_layer_activation: nn.Module = nn.LeakyReLU,
        move_transition_scale: float = 10.0,
        max_allowed_loss: float = +torch.inf,
        learning_rate: float = 1e-2,
    ):
        if obs_mask is None:
            obs_mask = torch.ones((obs_size,), dtype=torch.bool)
        
        self.obs_mask = obs_mask
        self.optimize_frequency = mini_batch_size
        self.use_sigmoid_output = use_sigmoid_output
        self.move_transition_scale = move_transition_scale

        self.network = PlayerModelNetwork(
            input_dim=obs_size,
            output_dim=n_moves,
            use_sigmoid_output=self.use_sigmoid_output,
            input_clip=input_clip,
            input_clip_leaky_coef=input_clip_leaky_coef,
            hidden_layer_sizes=hidden_layer_sizes,
            hidden_layer_activation=hidden_layer_activation,
        )

        # Mean squared error
        if self.use_sigmoid_output:
            self.loss_function = lambda predicted, target: torch.mean((predicted - target)**2, dim=1)
        # Cross entropy loss
        else:
            self.loss_function = lambda predicted, target: -torch.sum(torch.log(predicted) * target, dim=1)

        self.optimizer = torch.optim.SGD(params=self.network.parameters(), lr=learning_rate)

        # Just to make training easier, know which layers actually have learnable parameters
        self.learnable_layer = [
            ("weight" in param_names and "bias" in param_names)
            for param_names in map(
                lambda params: map(lambda t: t[0], params),
                map(
                    list,
                    map(
                        nn.Module.named_parameters,
                        self.network.layers
                    ),
                )
            )
        ]

        # Maximum allowed loss before proceeding with the next training examples
        self.max_loss = max_allowed_loss

        self.x_batch_as_list = []
        self.y_batch_as_list = []
        self.step = 0
        # Whether the first observation in the training batch corresponds to a move transition in the part of the player.
        # Important when determining the move transtion scale for each observation
        self.first_obs_of_batch_had_move_transition = None
        self.cummulative_loss = 0
        self.cummulative_loss_n = 0

    def mask_environment_observation(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, self.obs_mask]

    def update(self, obs: torch.Tensor, action: torch.Tensor, move_transition: bool = False):
        self.x_batch_as_list.append(obs)
        self.y_batch_as_list.append(action)
        self.step += 1
        if self.first_obs_of_batch_had_move_transition is None:
            self.first_obs_of_batch_had_move_transition = move_transition

        if self.step >= self.optimize_frequency:
            batch_x = torch.cat(self.x_batch_as_list, 0)
            batch_y = torch.cat(self.y_batch_as_list, 0)
            loss = None
            base_lr = self.optimizer.param_groups[0]["lr"]
            lr_modifier = 1

            move_transition_multiplier = 1 + torch.hstack((torch.tensor(self.first_obs_of_batch_had_move_transition), torch.any(batch_x[:-1, 2:32] != batch_x[1:, 2:32], dim=1))) * (self.move_transition_scale - 1)

            # Keep training on examples that are problematic
            while loss is None or loss > self.max_loss:
                self.optimizer.param_groups[0]["lr"] = base_lr * lr_modifier
                self.optimizer.zero_grad()

                predicted = self.network(batch_x)
                loss = torch.mean(self.loss_function(predicted, batch_y) * move_transition_multiplier)

                loss.backward()
                self.optimizer.step()

                self.cummulative_loss += loss.item()
                self.cummulative_loss_n += 1

                # Investigate learning is dead problem which is occurring with current mimic model (even with leaky net)
                if all(
                    not torch.any(layer.weight.grad) and not torch.any(layer.bias.grad)
                    for i, layer in enumerate(self.network.layers)
                    if self.learnable_layer[i]
                ):
                    raise RuntimeError(
                        f"learning is dead, gradients are 0! (loss: {loss.item()})"
                    )

            self.optimizer.param_groups[0]["lr"] = base_lr
            self.x_batch_as_list.clear()
            self.y_batch_as_list.clear()
            self.step = 0
            self.first_obs_of_batch_had_move_transition = None

    def predict(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        snap_to_fraction_of: int = 10,
    ) -> "any":
        with torch.no_grad():
            probs = self.probability_distribution(obs, snap_to_fraction_of)

            if deterministic:
                return torch.argmax(probs, axis=1)
            else:
                probs_cummulative = torch.cumsum(probs, 1)
                # Contrary to what the docs say, torch.rand can return 1..., so we multiply by 0.99 to compensate for that
                sampled = torch.rand(probs_cummulative.shape[0], 1) * 0.99
                # I just want to know the index of the first time sampled < probs_cummulative on each row, but it's convoluted
                nonzeros = torch.nonzero(sampled <= probs_cummulative)
                # Information across all rows is in a single dimension, so we need to determine the points in which we transition from one row to the next
                i = torch.squeeze(
                    torch.nonzero((nonzeros[:-1] - nonzeros[1:])[:, 0]) + 1, dim=1
                )
                # Add the initial index that is not taken into account by the previous operation
                i = torch.cat((torch.tensor([0]), i))
                try:
                    return nonzeros[i, 1]
                except IndexError as e:
                    print("wrong indexing")
                    print(" nonzeros:", nonzeros)
                    print(" i:", i)
                    print(" obs:", obs)
                    print(" probs:", probs)
                    print(" sampled:", sampled)
                    print(" probs_cummulative:", probs_cummulative)
                    raise e

    # TODO: if over primitive actions, the model is prone to having great uncertainty (why, past me?)
    def probability_distribution(
        self, obs: torch.Tensor, snap_to_fraction_of: int = 10
    ):
        with torch.no_grad():
            out = self.network(obs)
            if self.use_sigmoid_output:
                snapped = torch.round(snap_to_fraction_of * out) / snap_to_fraction_of
                if torch.all(snapped == 0):
                    print(
                        f"Oops, one distribution had all 0s ({out})! Will use uniform distribution"
                    )
                    probs = torch.ones(snapped.shape) / snapped.shape[1]
                else:
                    probs = snapped / torch.sum(snapped, axis=1, keepdim=True)
            
            else:
                probs = out

        return probs

    def load(self, path: str):
        self.network.load_state_dict(torch.load(path))

    def save(self, path: str):
        torch.save(self.network.state_dict(), path)


# TODO: there are three different ways we can predict actions: primitive actions discretized, primitive actions as tuple, and moves. Are all supported?
class FootsiesAgent(FootsiesAgentBase):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        frameskipping: bool = True,
        append_time_since_last_interaction: bool = False,
        by_primitive_actions: bool = False,
        use_sigmoid_output: bool = False,
        input_clip: bool = False,
        input_clip_leaky_coef: float = 0,
        hidden_layer_sizes_specification: str = "64,64",
        hidden_layer_activation_specification: str = "LeakyReLU",
        move_transition_scale: float = 10.0,
        max_allowed_loss: float = +torch.inf,
        mini_batch_size: int = 1,
        learning_rate: float = 1e-2,
        learn_p1: bool = True,
        learn_p2: bool = True,
    ):
        if append_time_since_last_interaction:
            raise NotImplementedError("appending the time since the last interaction is still not supported")

        self.frameskipping = frameskipping
        self.by_primitive_actions = by_primitive_actions
        self.learn_p1 = learn_p1
        self.learn_p2 = learn_p2

        self.n_moves = action_space.n if by_primitive_actions else ActionMap.n_simple()
        
        # Masks to selectively remove the portion of the observation that is not relevant (occurs if frameskipping, it is the player's move progress)
        p1_observation_mask = torch.ones((observation_space.shape[0],), dtype=torch.bool)
        p2_observation_mask = torch.ones((observation_space.shape[0],), dtype=torch.bool)
        if self.frameskipping:
            p1_observation_mask[32] = False
            p2_observation_mask[33] = False

        # Both models have mostly the same structure and training regime
        player_model_kwargs = {
            # We don't consider the move progress of the player themselves (e.g. if I'm modeling player 2, then I don't need to know its move progress)
            "obs_size": observation_space.shape[0] - (1 if self.frameskipping else 0),
            "n_moves": self.n_moves,
            "mini_batch_size": mini_batch_size,
            "use_sigmoid_output": use_sigmoid_output,
            "input_clip": input_clip,
            "input_clip_leaky_coef": input_clip_leaky_coef,
            "hidden_layer_sizes": [int(n) for n in hidden_layer_sizes_specification.split(",")] if hidden_layer_sizes_specification else [],
            "hidden_layer_activation": getattr(nn, hidden_layer_activation_specification),
            "move_transition_scale": move_transition_scale,
            "max_allowed_loss": max_allowed_loss,
            "learning_rate": learning_rate,
        }
        self.p1_model = PlayerModel(
            **player_model_kwargs,
            obs_mask=p1_observation_mask,
        )
        self.p2_model = PlayerModel(
            **player_model_kwargs,
            obs_mask=p2_observation_mask,
        )

        self.previous_observation = None
        self.previous_p1_move_state: FootsiesMove = None
        self.previous_p2_move_state: FootsiesMove = None

        self._test_observations = None
        self._test_actions = None
        self._p1_correct = 0
        self._p2_correct = 0
        self._random_correct = 0
        # Multiply the "correct" counters by the decay to avoid infinitely increasing counters and prioritize recent values.
        # Set to a value such that the 1000th counter value in the past will have a weight of 1%
        self._correct_decay = 0.01 ** (1 / 1000)

    def act(self, obs, info: dict, p1: bool = True, deterministic: bool = False, predict: bool = False) -> "any":
        obs = self._obs_to_tensor(obs)
        self.previous_observation = obs
        self.previous_p1_move_state = FOOTSIES_MOVE_INDEX_TO_MOVE[info["p1_move"]]
        self.previous_p2_move_state = FOOTSIES_MOVE_INDEX_TO_MOVE[info["p2_move"]]

        if predict:
            model = self.p1_model if p1 else self.p2_model
            prediction = model.predict(model.mask_environment_observation(obs), deterministic=deterministic).item()
            return prediction

        return 0

    def update(
        self, next_obs, reward: float, terminated: bool, truncated: bool, info: dict
    ):
        key = "action" if self.by_primitive_actions else "move"
        
        # The move_state variables are the direct FOOTSIES moves, all other variables correspond to "action moves" as specified in the utils
        p1_move_state: FootsiesMove = FOOTSIES_MOVE_INDEX_TO_MOVE[info["p1_" + key]]
        p2_move_state: FootsiesMove = FOOTSIES_MOVE_INDEX_TO_MOVE[info["p2_" + key]]
        p1_simple = ActionMap.simple_from_move(p1_move_state)
        p2_simple = ActionMap.simple_from_move(p2_move_state)
        p1_simple_oh = self._move_onehot(p1_simple)
        p2_simple_oh = self._move_onehot(p2_simple)

        p1_observation = self.p1_model.mask_environment_observation(self.previous_observation)
        p2_observation = self.p2_model.mask_environment_observation(self.previous_observation)

        if self.learn_p1 and (not self.frameskipping or ActionMap.is_state_actionable_late(self.previous_p1_move_state, self.previous_observation[0, 32], next_obs[32])):
            self.p1_model.update(p1_observation, p1_simple_oh, self.previous_p1_move_state != p1_move_state)
        if self.learn_p2 and (not self.frameskipping or ActionMap.is_state_actionable_late(self.previous_p2_move_state, self.previous_observation[0, 33], next_obs[33])):
            self.p2_model.update(p2_observation, p2_simple_oh, self.previous_p2_move_state != p2_move_state)

        # Update metrics
        # self._p1_correct = (self._p1_correct * self._correct_decay) + (
        #     self.p1_model.predict(p1_observation) == p1_move_idx
        # )
        # self._p2_correct = (self._p2_correct * self._correct_decay) + (
        #     self.p2_model.predict(p2_observation) == p2_move_idx
        # )
        # self._random_correct = (self._random_correct * self._correct_decay) + (
        #     torch.rand((1,)).item() < 1 / self.n_moves
        # )

    def _obs_to_tensor(self, obs):
        return torch.tensor(obs, dtype=torch.float32).reshape((1, -1))

    def _move_onehot(self, move: int):
        return nn.functional.one_hot(torch.tensor(move), num_classes=self.n_moves).unsqueeze(0)

    def decision_entropy(self, obs: torch.Tensor, p1: bool) -> float:
        model = self.p1_model if p1 else self.p2_model

        probs = model.probability_distribution(model.mask_environment_observation(obs))

        return torch.nansum(probs * torch.log2(probs))

    def load(self, folder_path: str):
        p1_path = os.path.join(folder_path, "p1")
        p2_path = os.path.join(folder_path, "p2")
        self.p1_model.load(p1_path)
        self.p2_model.load(p2_path)

    def save(self, folder_path: str):
        p1_path = os.path.join(folder_path, "p1")
        p2_path = os.path.join(folder_path, "p2")
        self.p1_model.save(p1_path)
        self.p2_model.save(p2_path)

    def evaluate_average_loss_and_clear(self, p1: bool) -> float:
        model = self.p1_model if p1 else self.p2_model

        res = (
            (model.cummulative_loss / model.cummulative_loss_n)
            if model.cummulative_loss_n != 0
            else 0
        )

        model.cummulative_loss = 0
        model.cummulative_loss_n = 0
        return res

    def evaluate_performance(self, p1: bool) -> float:
        """Evaluate relative performance against a random predictor (how many times does it predict better than a random predictor)"""
        correct = self._p1_correct if p1 else self._p2_correct
        return correct / self._random_correct

    def _initialize_test_states(self, test_states: List[torch.Tensor]):
        if self._test_observations is None or self._test_actions is None:
            # If the state is terminal, then there will be no action. Discard those states
            observations, actions = map(
                np.array, zip(*filter(lambda sa: sa[1] is not None, test_states))
            )
            self._test_observations = torch.tensor(observations, dtype=torch.float32)
            self._test_actions = torch.tensor(actions, dtype=torch.float32)

    def evaluate_divergence_between_players(
        self, test_states: List[torch.Tensor]
    ) -> float:
        self._initialize_test_states(test_states)

        p1_predicted = self.p1_model.predict(
            self.p1_model.mask_environment_observation(self._test_observations), deterministic=False
        )
        p2_predicted = self.p2_model.predict(
            self.p2_model.mask_environment_observation(self._test_observations), deterministic=False
        )

        return torch.sum(p1_predicted == p2_predicted) / self._test_actions.size(0)

    def evaluate_decision_entropy(
        self, test_states: List[torch.Tensor], p1: bool
    ) -> float:
        self._initialize_test_states(test_states)

        return self.decision_entropy(self._test_observations, p1=p1)

    # NOTE: this only works if the class was defined to be over primitive actions
    def extract_policy(
        self, env: Env, use_p1: bool = True
    ) -> Callable[[dict], Tuple[bool, bool, bool]]:
        model_to_use = self.p1_model if use_p1 else self.p2_model
        obs_mask = self.p1_model.obs_mask if use_p1 else self.p2_model.obs_mask

        policy_network = deepcopy(model_to_use.network)
        policy_network.requires_grad_(False)

        def internal_policy(obs):
            obs = self._obs_to_torch(obs)
            predicted = policy_network(obs[:, obs_mask])
            return torch.argmax(predicted)

        return super()._extract_policy(env, internal_policy)
