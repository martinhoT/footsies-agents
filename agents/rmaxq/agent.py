from collections import defaultdict
from agents.base import FootsiesAgentBase
import numpy as np


# TODO: check https://github.com/mlanden/RMAXQ for alternate implementation
# TODO: other HRL implementations: https://github.com/Kirili4ik/HRL-taxi and https://github.com/aijunbai/taxi
# TODO: the model (transition and reward functions) can be simplified if we assume it to be deterministic
class FootsiesAgent(FootsiesAgentBase):
    def __init__(self):
        self.reward_counters = defaultdict(lambda: 0)
        self.action_pair_counters = defaultdict(lambda: 0)
        self.transition_counters = defaultdict(lambda: 0)
        self.t = 0
        self.timestamp = defaultdict(lambda: 0)
        self.envelope: "dict[any, set]" = {}
        self.q_table = defaultdict(dict)
        self.v_table = defaultdict(dict)
        self.policy = defaultdict(dict)
        self.transition_probability = defaultdict(dict)
        self.reward = defaultdict(dict)
        
        # Set containing all possible environment states
        # TODO: how do we define the set of all states?? Surely we can't contain them all...
        self.states = set()
        
        # Threshold sample size
        # TODO: define a value for this
        self.m = None
    
    def rmaxq(self, obs, action):
        # Action is primitive
        if self.is_primitive(action):
            next_obs, reward, done = _execute_(action)
            
            # Record primitive data
            self.reward_counters[obs, action] += reward
            self.action_pair_counters[obs, action] += 1
            self.transition_counters[obs, action, next_obs] += 1
            self.t += 1

            return next_obs, done
        
        # Action is composite
        else:
            # Until the observed state is terminal for this action or the episode terminates
            while obs not in self.terminal_states(action) and not done:
                self.compute_policy(obs, action)
                # Recursive execution
                obs, done = self.rmaxq(self.policy(obs, of_action=action), obs)

            return obs

    def compute_policy(self, obs, action):
        if self.timestamp[action] < self.t:
            self.timestamp[action] = self.t
            self.envelope[action] = set()

        self.prepare_envelope(obs, action)

        # Value iteration
        while not _converged_():
            for next_obs in self.envelope[action]:
                for child_action in self.child_actions(action, at_obs=next_obs):
                    self.update_q_table(action, next_obs, child_action)
                
                self.update_v_table(action, next_obs)
        
        # Importing numpy only for argmax wow
        self.policy[action][obs] = np.argmax([self.q_table[action][obs, a] for a in self.child_actions(action, obs)])

    def prepare_envelope(self, obs, action):
        if obs not in self.envelope[action]:
            self.envelope[action].add(obs)

            for child_action in self.child_actions(action, obs):
                self.compute_model(obs, child_action)
                for next_obs in self.next_possible_states(action, obs, next_obs):
                    self.prepare_envelope(next_obs, action)

    def compute_model(self, obs, action):
        # Action is primitive
        if self.is_primitive(action):
            if self.action_pair_counters[obs, action] >= self.m:
                self.reward[action][obs] = self.reward_counters[obs, action] / self.action_pair_counters[obs, action]
                for next_obs in self.states:
                    self.transition_probability[action][obs, next_obs] = self.transition_counters[obs, action, next_obs] / self.action_pair_counters[obs, action]
        
        # Action is composite
        else:
            self.compute_policy(obs, action)
            
            # Dynamic programming
            while not _converged_():
                for next_obs in self.envelope[action]:
                    self.update_reward(action, next_obs)
                    for terminal_state in self.terminal_states(action):
                        self.update_transition_probability(action, next_obs, terminal_state)

    def is_primitive(self, action):
        raise NotImplementedError
    
    def terminal_states(self, action):
        raise NotImplementedError

    def policy(self, obs, of_action):
        raise NotImplementedError

    def child_actions(self, of_action, at_obs):
        raise NotImplementedError

    def goal_reward(self, of_action, at_obs):
        raise NotImplementedError

    # NOTE: this can be heavily simplified for a deterministic environment
    def next_possible_states(self, action, obs, next_obs) -> set:
        return {s for s in self.states if self.transition_probability[action][obs, next_obs] > 0}

    # Dynamic programming required
    def update_q_table(self, action, child_action, next_obs):
        """Implementation of equation 1"""
        raise NotImplementedError

    def update_v_table(self, action, obs):
        """Implementation of equation 2"""
        if obs in self.terminal_states(action):
            self.v_table[action][obs] = self.goal_reward(action, obs)
        else:
            self.v_table[action][obs] = max(self.q_table[action][obs, a] for a in self.child_actions(action, obs))

    def update_reward(self, action, obs):
        """Implementation of equation 4"""
        raise NotImplementedError

    def update_transition_probability(self, action, obs, next_obs):
        """Implementation of equation 5"""
        raise NotImplementedError

    def act(self, obs) -> "any":
        ...

    def update(self, next_obs, reward: float, terminated: bool, truncated: bool):
        ...

    def load(self, folder_path: str):
        ...

    def save(self, folder_path: str):
        ...
