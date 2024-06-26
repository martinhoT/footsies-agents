import os
import torch as T
import random
import logging
from torch import nn
from torch.distributions import Categorical
from agents.action import ActionMap
from agents.base import FootsiesAgentBase
from agents.a2c.a2c import A2CQLearner, ValueNetwork
from agents.logger import TestState
from agents.ql.ql import QFunction, QFunctionTable, QFunctionNetwork
from agents.torch_utils import AggregateModule

LOGGER = logging.getLogger("main.a2c.agent")


class A2CAgent(FootsiesAgentBase):
    
    def __init__(
        self,
        learner: A2CQLearner,
        opponent_action_dim: int,
        consider_explicit_opponent_policy: bool = False,
        act_with_qvalues: bool = False,
        one_decision_at_hitstop: bool = True,
        will_act_anyway: bool = False,
    ):
        """
        Footsies agent using the A2C algorithm, potentially with some modifications.
        This implementation assumes simplified actions.

        Parameters
        ----------
        - `opponent_action_dim`: the number of opponent actions
        - `learner`: the A2C algorithm class to use. If `None`, one will be created
        Only valid for FOOTSIES, since it's an environment with 2 players on the same conditions (characters, more specifically)
        - `consider_explicit_opponent_policy`: whether to calculate critic values assuming an explicit opponent policy.
        The opponent policy is specified in the `next_opponent_policy` key of the `info` dictionary in `update`.
        - `act_with_qvalues`: whether to act according to a softmax over Q-values, instead of using the actor's policy.
        """
        self.opponent_action_dim = opponent_action_dim
        self.consider_explicit_opponent_policy = consider_explicit_opponent_policy
        self._act_with_qvalues = act_with_qvalues
        self._one_decision_at_hitstop = one_decision_at_hitstop
        self._will_act_anyway = will_act_anyway

        self._learner = learner
        self._actor = learner.actor
        self._critic = learner.critic
        
        modules: dict[str, nn.Module] = {"actor": self._actor}
        if isinstance(self._critic, ValueNetwork):
            modules["critic"] = self._critic
        elif isinstance(self._critic, QFunctionNetwork):
            modules["critic"] = self._critic.q_network
        self._model = AggregateModule(modules)

        # The action that the agent *consciously* attempted.
        # When the agent can't do anything this is None.
        self._current_action: int = 0
        # Whether the agent has performed an action during hitstop.
        # While in hitstop, time is frozen, so the agent should only perform one action.
        # Likewise, there should only be one update, from the moment the agent performs an action to the moment hitstop ends.
        # We have two variables, one for act() and another for update(), since with reaction time they may be de-synced
        # and learning (update) should be independent of reaction time
        # (especially since processing of this variable may be wrong as 'obs' may be corrected observation).
        self._has_acted_in_hitstop_act: bool = False
        self._has_acted_in_hitstop_update: bool = False

        # For logging
        self.cumulative_delta = 0
        self.cumulative_delta_n = 0
        self.cumulative_qtable_error = 0
        self.cumulative_qtable_error_n = 0
        self._test_observations = None

    # NOTE: reminder, we should use `info` carefully here, since when using reaction time it won't be delayed
    def act(self, obs: T.Tensor, info: dict, predicted_opponent_action: int | None = None, deterministic: bool = False) -> int:
        # If we can't perform an action, don't even attempt one.
        # Unless we just don't care and act anyway.
        agent_simple_completed = info["agent_simple_completed"]
        if self._will_act_anyway:
            if not agent_simple_completed:
                return 0
        elif self.wont_act(agent_simple_completed, obs, self._has_acted_in_hitstop_act):
            return 0
        
        # If we have passed the above condition, then it is because we are free to act, so we haven't acted yet.
        self._has_acted_in_hitstop_act = False

        # NOTE: this means that by default, without an opponent model, we assume the opponent is uniform random, which is unrealistic.
        if predicted_opponent_action is None:
            predicted_opponent_action = random.randint(0, self.opponent_action_dim - 1)

        if self.act_with_qvalues:
            qs = self._learner.critic.q(obs, opponent_action=predicted_opponent_action).detach()

            if deterministic:
                simple_action = qs.argmax().item()
            else:
                dist = Categorical(probs=nn.functional.softmax(qs, dim=-1))
                simple_action = dist.sample().item()

        else:
            if deterministic:
                simple_action = self._learner.actor.probabilities(obs, next_opponent_action=predicted_opponent_action).argmax().item()
            else:
                simple_action = self._learner.sample_action(obs, next_opponent_action=predicted_opponent_action)
        
        action = int(simple_action)
        self._current_action = action

        # Update the respective variable.
        # This basically means we do not want to act further until we are back in a neutral state.
        is_in_hitstop = ActionMap.is_in_hitstop_torch(obs)
        if is_in_hitstop:
            self._has_acted_in_hitstop_act = True

        return action

    def update(self, obs: T.Tensor, next_obs: T.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict):
        # We always consider the agent's simple action, never the one inferred from the observation.
        # We cannot use "agent_simple" from info because it considers the inaction ones (the deterministic 0s in act) as simple actions.
        obs_agent_action = self._current_action
        obs_opponent_action = next_info["p2_simple"]

        wont_act_obs = self.wont_act(info["agent_simple_completed"], obs, self._has_acted_in_hitstop_update)
        if not wont_act_obs:
            # If we have passed the condition, then it is because we were free to act, so we have acted.
            # If we were in hitstop, then we surely acted in hitstop.
            self._has_acted_in_hitstop_update = bool(ActionMap.is_in_hitstop_torch(obs).item())

        next_opponent_policy = next_info.get("next_opponent_policy", None) if self.consider_explicit_opponent_policy else None
        if next_opponent_policy is not None:
            next_opponent_policy = next_opponent_policy.unsqueeze(1)

        self._learner.learn(obs, next_obs, reward, terminated, truncated,
            obs_agent_action=obs_agent_action,
            obs_opponent_action=obs_opponent_action,
            agent_will_frameskip=self.wont_act(next_info["agent_simple_completed"], next_obs, self._has_acted_in_hitstop_update),
            opponent_will_frameskip=not next_info["p2_is_actionable"],
            next_obs_opponent_policy=next_opponent_policy,
            intrinsic_reward=next_info.get("intrinsic_reward", 0),
        )

        # P1 learning logging
        if isinstance(self._learner, A2CQLearner):
            if self._learner.delta is not None:     
                self.cumulative_delta += self._learner.delta
                self.cumulative_delta_n += 1
            if self._learner.extrinsic_td_error is not None:
                self.cumulative_qtable_error += self._learner.extrinsic_td_error
                self.cumulative_qtable_error_n += 1
        
        if terminated or truncated:
            self._current_action = 0
            self._has_acted_in_hitstop_act = False
            self._has_acted_in_hitstop_update = False
    
    def wont_act(self, completed_previous_action: bool, obs: T.Tensor, has_acted_in_hitstop: bool) -> bool:
        """Whether the agent will attempt to perform any action at the current state."""
        if not completed_previous_action:
            return True

        is_at_neutral_actionable = ActionMap.is_at_neutral_actionable_torch(obs)
        if (self._one_decision_at_hitstop and has_acted_in_hitstop and not is_at_neutral_actionable):
            return True

        is_state_actionable = ActionMap.is_state_actionable_torch(obs)
        return not is_state_actionable

    @property
    def model(self) -> nn.Module:
        return self._model
    
    @property
    def learner(self) -> A2CQLearner:
        return self._learner

    @property
    def current_action(self) -> int:
        return self._current_action

    @property
    def act_with_qvalues(self) -> bool:
        return self._act_with_qvalues

    @act_with_qvalues.setter
    def act_with_qvalues(self, value: bool):
        self._act_with_qvalues = value

    # Need to use custom save and load functions because we could use a tabular Q-function
    def load(self, folder_path: str):
        # Load actor
        actor_path = os.path.join(folder_path, "actor")
        self._actor.load_state_dict(T.load(actor_path))
        
        # Load critic
        if isinstance(self._critic, QFunctionTable):
            critic_path = os.path.join(folder_path, "critic_qtable")
            self._critic.load(critic_path)
        elif isinstance(self._critic, QFunctionNetwork):
            critic_path = os.path.join(folder_path, "critic_qnetwork")
            self._critic.load(critic_path)
        elif isinstance(self._critic, ValueNetwork):
            critic_path = os.path.join(folder_path, "critic_vnetwork")
            self._critic.load_state_dict(T.load(critic_path))

    def save(self, folder_path: str):
        # Save actor
        actor_path = os.path.join(folder_path, "actor")
        T.save(self._actor.state_dict(), actor_path)
        
        # Save critic
        def save_critic(critic: QFunction, folder_path: str):
            if isinstance(critic, QFunctionTable):
                critic_path = os.path.join(folder_path, "critic_qtable")
                critic.save(critic_path)
            elif isinstance(critic, QFunctionNetwork):
                critic_path = os.path.join(folder_path, "critic_qnetwork")
                critic.save(critic_path)
            elif isinstance(critic, ValueNetwork):
                critic_path = os.path.join(folder_path, "critic_vnetwork")
                T.save(critic.state_dict(), critic_path)
        
        save_critic(self._critic, folder_path)
        if self.learner.intrinsic_critic is not None:
            save_critic(self.learner.intrinsic_critic, folder_path)

    def evaluate_average_delta(self) -> float:
        res = (
            self.cumulative_delta / self.cumulative_delta_n
        ) if self.cumulative_delta_n != 0 else 0

        self.cumulative_delta = 0
        self.cumulative_delta_n = 0

        return res
        
    def evaluate_average_qtable_error(self) -> float:
        res = (
            self.cumulative_qtable_error / self.cumulative_qtable_error_n
        ) if self.cumulative_qtable_error_n != 0 else 0

        self.cumulative_qtable_error = 0
        self.cumulative_qtable_error_n = 0

        return res

    def _initialize_test_states(self, test_states: list[TestState]):
        if self._test_observations is None:
            test_observations = [s.observation for s in test_states]
            self._test_observations = T.vstack(test_observations)

    def evaluate_average_policy_entropy(self, test_states: list[TestState]) -> float:
        self._initialize_test_states(test_states)

        with T.no_grad():
            probs = self._actor(self._test_observations)
            entropies = -T.sum(T.log(probs + 1e-8) * probs, dim=-1)
            return T.mean(entropies).item()
