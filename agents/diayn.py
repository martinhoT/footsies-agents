import torch
import gymnasium
from torch import nn
from agents.torch_utils import create_layered_network
from typing import Any
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


class DiscriminatorNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        skill_dim: int,
        hidden_layer_sizes: list[int],
        hidden_layer_activation: type[nn.Module] = nn.ReLU,
    ):
        super().__init__()

        self.layers = create_layered_network(observation_dim, skill_dim, hidden_layer_sizes, hidden_layer_activation)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(observation)

    def probability(self, observation: torch.Tensor) -> torch.Tensor:
        """Get output as probabilities, by application of the softmax function."""
        return self.softmax(self(observation))


class DIAYN:
    def __init__(
        self,
        observation_dim: int,
        skill_dim: int,
        include_baseline: bool = True,
        discriminator_learning_rate: float = 1e-3,
        discriminator_hidden_layer_sizes: list[int] = None,
        discriminator_hidden_layer_activation: type[nn.Module] = nn.ReLU,
    ):
        """
        Implementation of the DIAYN (Diversity is All You Need) algorithm, for learning a set of task-agnostic skills.
        Including the discriminator network, but not any high-level controller for choosing skills.

        An instance should be used in a training loop as follows (assuming the Gymnasium API):
        ```python
        diayn = DIAYN(...)
        while not converged:
            skill = diayn.sample_skill_uniform()
            obs, _ = env.reset()
            while not (terminated or truncated):
                action = policy(obs, skill)
                obs, _, terminated, truncated, _ = env.step(action)
                # Task reward is discarded in favor of pseudo-reward
                reward = diayn.pseudo_reward(obs)
                # Update the agent's policy to maximize pseudo-reward
                ...
                diayn.update_discriminator()
        ```

        Entropy regularization is handled by the policy.
        In the original work, the Soft Actor-Critic (SAC) algorithm already handled this case.
        The regularizer was scaled by `0.1`.

        Parameters
        ----------
        - `observation_dim`: the dimensionality of the environment's observations
        - `skill_dim`: the dimensionality of the skill vectors, roughly equivalent to the desired number of skills to learn
        - `include_baseline`: whether to include the baseline `-log(z)` in the pseudo-reward.
        Including the baseline makes the pseudo-reward non-negative, incentivizing the agent to stay alive.
        Not including it incentivizes the agent to end the episode as soon as possible, which might be useful for some environments (such as mountain car)
        - `discriminator_learning_rate`: the learning rate for the discriminator
        - `discriminator_hidden_layer_sizes`: the sizes of the hidden layers of the discriminator
        - `discriminator_hidden_layer_activation`: the activation function to use in the hidden layers of the discriminator
        """

        self.skill_dim = skill_dim
        self.include_baseline = include_baseline
        
        self.discriminator = DiscriminatorNetwork(
            observation_dim=observation_dim,
            skill_dim=skill_dim,
            hidden_layer_sizes=discriminator_hidden_layer_sizes,
            hidden_layer_activation=discriminator_hidden_layer_activation,
        )

        self.discriminator_loss_fn = nn.CrossEntropyLoss()
        self.discriminator_optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=discriminator_learning_rate)

        # NOTE: the base of the distribution's entropy is e
        self.uniform_skill_distribution = torch.distributions.OneHotCategorical(probs=torch.ones(skill_dim) / skill_dim)

        self._current_skill = None
        self._current_observation = None
        
        # For tracking purposes
        self._discriminator_last_loss = None
        self._last_skill_probs = None

    def sample_skill_uniform(self) -> torch.Tensor:
        """
        Randomly samples a skill uniformly from a categorical distribution of skills. This should be done at the beginning of each episode.
        
        NOTE: the sampled skill is stored.
        """
        self._current_skill = self.uniform_skill_distribution.sample().unsqueeze(0)
        return self._current_skill

    def pseudo_reward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Calculate the pseudo-reward to feed to the agent.
        
        NOTE: the provided observation is stored.
        """
        self._current_observation = observation
        skill_probs = self.discriminator.probability(observation)
        discriminator_log_prob = torch.log(skill_probs[:, self.current_skill_id])

        reward = discriminator_log_prob
        if self.include_baseline:
            # We set the prior skill distribution p(z) to be uniform
            reward -= torch.log(1 / self.skill_dim)
        
        self._last_skill_probs = skill_probs

        return reward

    def update_discriminator(self):
        """Update the discriminator to better distinguish between different skills."""
        self.discriminator_optimizer.zero_grad()
        loss = self.discriminator_loss_fn(self.discriminator(self._current_observation), self._current_skill)
        loss.backward()
        self.discriminator_optimizer.step()

        self._discriminator_last_loss = loss.item()

    @property
    def effective_number_of_skills(self) -> float:
        """The effective number of skills, which is the exponential of the entropy of the skill distribution."""
        return torch.exp(self.uniform_skill_distribution.entropy()).item()

    @property
    def current_skill(self) -> torch.Tensor:
        return self._current_skill

    @property
    def current_skill_id(self) -> int:
        return torch.argmax(self._current_skill).item()
    
    @property
    def discriminator_last_entropy(self) -> float:
        return torch.distributions.Categorical(probs=self._last_skill_probs).entropy().item()

    @property
    def discriminator_last_loss(self) -> float:
        return self._discriminator_last_loss


class DIAYNWrapper(gymnasium.Wrapper):
    def __init__(
        self,
        env: gymnasium.Env,
        diayn: DIAYN,
        log_dir: str = None,
        log_frequency: int = 1000,
    ):
        """
        Wrapper on a Gymnasium environment to incorporate the DIAYN algorithm.

        Parameters
        ----------
        - `env`: the environment to wrap
        - `diayn`: the DIAYN instance to use
        - `log_dir`: the directory where to save the logs of the discriminator's training using Tensorboard. If `None`, then no logs are saved
        - `log_frequency`: the frequency with which to perform logging, in number of environment steps
        """
        super().__init__(env)

        self.env = env
        self.diayn = diayn

        self.observation_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(env.observation_space.shape[0] + diayn.skill_dim,))

        # Logging
        self.summary_writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None
        self.log_frequency = log_frequency
        self.current_step = 0

        # Tracking exponentially weighted averages
        self._discriminator_entropy_avg = 0.0
        self._discriminator_loss_avg = 0.0
        self._skill_rewards_avg = {skill: 0.0 for skill in range(self.diayn.skill_dim)}
        self._avg_factor = 1 - 1 / log_frequency

    def reset(self, *args, **kwargs) -> tuple[torch.Tensor, dict[str, Any]]:
        skill = self.diayn.sample_skill_uniform()
        print("Using skill", skill)
        obs, info = self.env.reset(*args, **kwargs)
        obs_with_skill = torch.hstack([obs, skill])
        return obs_with_skill, info
    
    def step(self, *args, **kwargs):
        obs, _, terminated, truncated, info = self.env.step(*args, **kwargs)
        # We need to detach these from the computation graph or else we will run out of memory!
        reward = self.diayn.pseudo_reward(obs).item()
        obs_with_skill = torch.hstack([obs, self.diayn.current_skill]).detach()
        
        self.diayn.update_discriminator()

        # Logging
        if self.summary_writer is not None:
            self._discriminator_entropy_avg = (self._avg_factor) * self._discriminator_entropy_avg + (1 - self._avg_factor) * self.diayn.discriminator_last_entropy
            self._discriminator_loss_avg = (self._avg_factor) * self._discriminator_loss_avg + (1 - self._avg_factor) * self.diayn.discriminator_last_loss
            self._skill_rewards_avg[self.diayn.current_skill_id] = (self._avg_factor) * self._skill_rewards_avg[self.diayn.current_skill_id] + (1 - self._avg_factor) * reward
            self.current_step += 1
            
            if self.current_step % self.log_frequency == 0:
                self.summary_writer.add_scalar(
                    "DIAYN/Discriminator entropy",
                    self._discriminator_entropy_avg,
                    self.current_step,
                )
                self.summary_writer.add_scalar(
                    "DIAYN/Discriminator loss",
                    self._discriminator_loss_avg,
                    self.current_step,
                )
                for skill, reward_avg in self._skill_rewards_avg:
                    if reward_avg is not None:
                        self.summary_writer.add_scalar(
                            f"DIAYN/Average reward of skill [{skill}]",
                            reward_avg,
                            self.current_step,
                        )
                        self._skill_rewards_avg[skill] = None

        return obs_with_skill, reward, terminated, truncated, info
