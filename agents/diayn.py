import torch
from torch import nn
from agents.torch_utils import create_layered_network


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
        skill_dim: int,
        include_baseline: bool = True,
        discriminator_learning_rate: float = 1e-3,
    ):
        """
        Implementation of the DIAYN (Diversity is All You Need) algorithm, for learning a set of task-agnostic skills, including the discriminator network.
        This class doesn't include any high-level controller for choosing skills.

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
        - `skill_dim`: the dimensionality of the skill vectors, roughly equivalent to the desired number of skills to learn
        - `include_baseline`: whether to include the baseline `-log(z)` in the pseudo-reward.
        Including the baseline makes the pseudo-reward non-negative, incentivizing the agent to stay alive.
        Not including it incentivizes the agent to end the episode as soon as possible, which might be useful for some environments (such as mountain car)
        - `discriminator_learning_rate`: the learning rate for the discriminator
        """

        self.skill_dim = skill_dim
        self.include_baseline = include_baseline
        
        self.discriminator = DiscriminatorNetwork()

        self.discriminator_loss_fn = nn.CrossEntropyLoss()
        self.discriminator_optimizer = torch.optim.SGD(self.discriminator.parameters(), lr=discriminator_learning_rate)

        # NOTE: the base of the distribution's entropy is e
        self.uniform_skill_distribution = torch.distributions.OneHotCategorical(probs=torch.ones(skill_dim) / skill_dim)

        self.current_skill = None

    def sample_skill_uniform(self) -> torch.Tensor:
        """
        Randomly samples a skill uniformly from a categorical distribution of skills. This should be done at the beginning of each episode.
        
        NOTE: the sampled skill is stored.
        """
        self.current_skill = self.uniform_skill_distribution.sample()
        return self.current_skill

    def pseudo_reward(self, observation: torch.Tensor) -> float:
        """
        Calculate the pseudo-reward to feed to the agent.
        
        NOTE: the provided observation is stored.
        """
        skill_id = torch.argmax(self.current_skill).item()
        discriminator_log_prob = torch.log(self.discriminator.probability(observation)[skill_id])
        
        reward = discriminator_log_prob
        if self.include_baseline:
            # We set the prior skill distribution p(z) to be uniform
            reward -= torch.log(1 / self.skill_dim)
        
        return reward

    def update_discriminator(self):
        """Update the discriminator to better distinguish between different skills."""
        self.discriminator_optimizer.zero_grad()
        loss = self.discriminator_loss_fn(self.discriminator(self.observation), self.current_skill)
        loss.backward()
        self.discriminator_optimizer.step()

    @property
    def effective_number_of_skills(self) -> float:
        """The effective number of skills, which is the exponential of the entropy of the skill distribution."""
        return torch.exp(self.uniform_skill_distribution.entropy()).item()

