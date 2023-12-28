from copy import deepcopy
import os
import numpy as np
import torch
from torch import nn
from agents.base import FootsiesAgentBase
from gymnasium import Env
from typing import Any, Callable, List, Tuple
from gymnasium import Space
from footsies_gym.moves import FootsiesMove


RELEVANT_MOVES = set(FootsiesMove) - {FootsiesMove.WIN, FootsiesMove.DEAD}


class InputClip(nn.Module):
    # Range of [5, 5] allows representing the sigmoid values of 0.01 and 0.99
    # The idea is that the network should not be too sure or unsure of the outcomes, and allow better adaptability by avoiding very small gradients at the sigmoid's tails
    # NOTE: watch out, gradients can become 0 without leaking, a leaky version has better adaptability (as in the Loss of Plasticity in Deep Continual Learning paper)
    def __init__(self, minimum: float = -5, maximum: float = 5, leaky_coef: float = 0):
        """Clip input into range"""
        super().__init__()
        self.minimum = minimum
        self.maximum = maximum
        self.leaky_coef = leaky_coef
    
    def forward(self, x: torch.Tensor):

        return torch.clip(x, min=self.minimum + self.leaky_coef * (x - 5), max=self.maximum + self.leaky_coef * (x + 5))


class ProbabilityDistribution(nn.Module):
    def __init__(self):
        """Makes input sum to 1"""
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x / torch.sum(x)


class DebugStoreRecent(nn.Module):
    def __init__(self):
        """Store the most recent input"""
        super().__init__()
        self.stored = None
    
    def forward(self, x: torch.Tensor):
        self.stored = x
        return x


# Sigmoid output is able to tackle action combinations.
# Also, softmax has the problem of not allowing more than one dominant action (for a stochastic agent, for instance), it only focuses on one of the inputs.
# Softmax also makes the gradients for each output neuron dependent on the values of the other output neurons
class PlayerModelNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, output_sigmoid: bool = False, leaky_coef: float = 0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            # nn.Linear(32, 32),
            # nn.LeakyReLU(),
            nn.Linear(32, output_dim),
        )

        self.debug_store_1 = DebugStoreRecent()
        self.debug_store_2 = DebugStoreRecent()
        self.debug_store_3 = DebugStoreRecent()

        if output_sigmoid:
            self.layers.append(InputClip(leaky_coef=leaky_coef))
            self.layers.append(self.debug_store_1)
            self.layers.append(nn.Sigmoid())
            self.layers.append(self.debug_store_2)
            # Since during training we are not giving the actual probability distributions as targets, it might not be appropriate to train assuming they are
            # self.layers.append(ProbabilityDistribution())
            # self.layers.append(self.debug_store_3)
        
        else:
            self.layers.append(nn.Softmax(dim=1))

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class PlayerModel:
    def __init__(
        self,
        obs_size: int,
        n_moves: int,
        optimize_frequency: int = 1000,
        distribution_noise: float = 1e-4,
        use_sigmoid: bool = True,
        leaky_coef: float = 0,
    ):
        self.optimize_frequency = optimize_frequency

        self.network = PlayerModelNetwork(obs_size, n_moves, output_sigmoid=use_sigmoid, leaky_coef=leaky_coef)
        # NOTE: CrossEntropyLoss already applies Softmax at the end, which is redundant
        # self.loss_function = nn.CrossEntropyLoss()
        # NOTE: when we don't have a probability distribution, don't use this, since 0s in the target don't have any effect compared to 1s.
        #       They would only have an effect if for every output neuron the values of the other output neurons mattered (as is the case when we are creating a probability distribution).
        # self.loss_function = lambda predicted, actual: torch.mean(
        #     -torch.sum(
        #         actual
        #         * torch.log(
        #             (predicted + distribution_noise)
        #             / torch.unsqueeze(
        #                 torch.sum(predicted + distribution_noise, axis=1), dim=1
        #             )
        #         ),
        #         axis=1,
        #     )
        # )
        self.loss_function = nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.SGD(params=self.network.parameters(), lr=1)
        
        # Maximum allowed loss before proceeding with the next training examples
        self.max_loss = +torch.inf

        self.x_batch_as_list = []
        self.y_batch_as_list = []
        self.step = 0
        self.cummulative_loss = 0
        self.cummulative_loss_n = 0

    def update(self, obs: torch.Tensor, action: torch.Tensor):
        self.x_batch_as_list.append(obs)
        self.y_batch_as_list.append(action)
        self.step += 1

        if self.step >= self.optimize_frequency:
            x_batch = torch.cat(self.x_batch_as_list)
            y_batch = torch.stack(self.y_batch_as_list)
            loss = None
            base_lr = self.optimizer.param_groups[0]["lr"]
            lr_modifier = 1

            # Keep training on examples that are problematic
            while loss is None or loss > self.max_loss:
                self.optimizer.param_groups[0]["lr"] = base_lr * lr_modifier
                self.optimizer.zero_grad()

                predicted = self.network(x_batch)
                loss = self.loss_function(predicted, y_batch)

                loss.backward()
                self.optimizer.step()

                self.cummulative_loss += loss.item()
                self.cummulative_loss_n += 1

                # TODO: investigate learning is dead problem which is occurring with current mimic model (even with leaky net)
                if all(
                    not torch.any(layer.weight.grad) and not torch.any(layer.bias.grad)
                    for layer in self.network.layers
                    if "weight" in layer.__dict__ and "bias" in layer.__dict__
                ):
                    raise RuntimeError("learning is dead, gradients are 0!")

            self.optimizer.param_groups[0]["lr"] = base_lr
            self.x_batch_as_list.clear()
            self.y_batch_as_list.clear()
            self.step = 0

    def predict(self, obs: torch.Tensor, deterministic: bool = False, snap_to_fraction_of: int = 10) -> "any":
        with torch.no_grad():
            probs = self.probability_distribution(obs, snap_to_fraction_of)
            
            if deterministic:
                return torch.argmax(probs, axis=1)
            else:    
                probs_cummulative = torch.cumsum(probs, 1)
                sampled = torch.rand((probs_cummulative.shape[0], 1))
                # I just want to know the index of the first time sampled < probs_cummulative on each row, but it's convoluted
                nonzeros = torch.nonzero(sampled < probs_cummulative)
                # Information across all rows is in a single dimension, so we need to determine the points in which we transition from one row to the next
                i = torch.squeeze(torch.nonzero((nonzeros[:-1] - nonzeros[1:])[:, 0]) + 1, dim=1)
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
                    raise e


    # TODO: if over primitive actions, the model is prone to having great uncertainty
    def probability_distribution(self, obs: torch.Tensor, snap_to_fraction_of: int = 10):
        with torch.no_grad():
            out = self.network(obs)
            snapped = torch.round(snap_to_fraction_of * out) / snap_to_fraction_of
            if torch.all(snapped == 0):
                print(f"Oops, one distribution had all 0s ({out})! Will use uniform distribution")
                probs = torch.ones(snapped.shape) / snapped.shape[1]
            else:
                probs = snapped / torch.sum(snapped, axis=1, keepdim=True)
            
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
        over_primitive_actions: bool = False,
        optimize_frequency: int = 1000,
        use_sigmoid: bool = True,
        leaky_coef: float = 0,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.over_primitive_actions = over_primitive_actions

        self.n_moves = action_space.n if over_primitive_actions else len(RELEVANT_MOVES)

        self.p1_model = PlayerModel(
            observation_space.shape[0], self.n_moves, optimize_frequency, use_sigmoid=use_sigmoid, leaky_coef=leaky_coef,
        )
        self.p2_model = PlayerModel(
            observation_space.shape[0], self.n_moves, optimize_frequency, use_sigmoid=use_sigmoid, leaky_coef=leaky_coef,
        )

        self.current_observation = None

        self._test_observations = None
        self._test_actions = None
        self._p1_correct = 0
        self._p2_correct = 0
        self._random_correct = 0
        # Multiply the "correct" counters by the decay to avoid infinitely increasing counters and prioritize recent values.
        # Set to a value such that the 1000th counter value in the past will have a weight of 1%
        self._correct_decay = 0.01 ** (1 / 1000)

    def act(self, obs, p1: bool = True, deterministic: bool = False) -> "any":
        model = self.p1_model if p1 else self.p2_model
        obs = self._obs_to_tensor(obs)
        self.current_observation = obs
        return model.predict(obs, deterministic=deterministic)

    def update(
        self, next_obs, reward: float, terminated: bool, truncated: bool, info: dict
    ):
        key = "action" if self.over_primitive_actions else "move"
        # This is assuming that, when using moves, all relevant moves have contiguous IDs from 0, without breaks
        # That is, the irrelevant moves (WIN and DEAD) have the largest IDs
        p1_move = info["p1_" + key]
        p2_move = info["p2_" + key]
        p1_move_oh = self._move_onehot(p1_move)
        p2_move_oh = self._move_onehot(p2_move)

        # TODO: evaluate only when a move transition occurs (essentially frame skipping)
        self.p1_model.update(self.current_observation, p1_move_oh)
        self.p2_model.update(self.current_observation, p2_move_oh)

        # Update metrics
        self._p1_correct = (self._p1_correct * self._correct_decay) + (self.p1_model.predict(self.current_observation) == p1_move)
        self._p2_correct = (self._p2_correct * self._correct_decay) + (self.p2_model.predict(self.current_observation) == p2_move)
        self._random_correct = (self._random_correct * self._correct_decay) + (torch.rand((1,)).item() < 1 / self.n_moves)

    def _obs_to_tensor(self, obs):
        return torch.tensor(obs, dtype=torch.float32).reshape((1, -1))

    def _move_onehot(self, move: int):
        onehot = torch.zeros((self.n_moves,))
        onehot[move] = 1
        return onehot

    # From https://en.wikipedia.org/wiki/Hick's_law#Law
    def decision_entropy(self, obs: torch.Tensor, p1: bool) -> float:
        model = self.p1_model if p1 else self.p2_model

        probs = model.probability_distribution(obs)
        return torch.sum(probs * torch.log2(1 / probs + 1))

    def load(self, folder_path: str):
        p1_path = os.path.join(folder_path, "p1")
        p2_path = os.path.join(folder_path, "p2")
        self.p1_model.save(p1_path)
        self.p2_model.save(p2_path)

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

    def evaluate_divergence_between_players(self, test_states: List[torch.Tensor]) -> float:
        self._initialize_test_states(test_states)

        p1_predicted = self.p1_model.predict(self._test_observations, deterministic=True)
        p2_predicted = self.p2_model.predict(self._test_observations, deterministic=True)

        return torch.sum(p1_predicted == p2_predicted) / self._test_actions.size(0)

    def evaluate_decision_entropy(self, test_states: List[torch.Tensor], p1: bool) -> float:
        self._initialize_test_states(test_states)

        return self.decision_entropy(self._test_observations, p1=p1)

    # NOTE: this only works if the class was defined to be over primitive actions
    def extract_policy(
        self, env: Env, use_p1: bool = True
    ) -> Callable[[dict], Tuple[bool, bool, bool]]:
        model_to_use = self.p1_model if use_p1 else self.p2_model

        policy_network = deepcopy(model_to_use.network)
        policy_network.requires_grad_(False)

        def internal_policy(obs):
            obs = self._obs_to_torch(obs)
            predicted = policy_network(obs)
            return torch.argmax(predicted)

        return super()._extract_policy(env, internal_policy)
