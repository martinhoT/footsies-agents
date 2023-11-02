from agents.base import FootsiesAgentBase
from typing import Any, Iterable
from collections import defaultdict
from gymnasium import Env


class FootsiesAgent(FootsiesAgentBase):
    def __init__(self):
        self.current_observation = None
        
        # Environment model
        self.transition_model = defaultdict(dict)
        self.reward_model = {}
        self.n_transitions = defaultdict(lambda: 0)
        self.n_action_pairs = defaultdict(lambda: 0)
        self.reward_action_pairs = defaultdict(lambda: 0)

    def act(self, obs) -> Any:
        self.current_observation = obs
        ...


    def update(self, obs, reward: float):
        next_observation = obs
        ...

    def preprocess(self, env: Env):
        def play_data_generator(e: Env):
            obs, info = e.reset()
            if "p1_action" not in info:
                raise ValueError("the environment doesn't support learning by imitation, expecting agent action to be in key 'p1_action' of info dict")

            for _ in ...:
                next_obs, reward, terminated, trunctated, info = e.step()

                yield obs, info["p1_action"], next_obs, reward
                obs = next_obs

                if terminated or trunctated:
                    obs, info = e.reset()
            
        self.learn_model(play_data_generator(env))

    # TODO: the model is actually updated bottom-up in the paper's implementation (start from primitive actions up to composed actions), check if it has implications
    def learn_model(self, play_data: "Iterable[tuple[Any, Any, Any, float]]"):
        """Learn the environment model by observing human play data (sequence of state, action, next state and reward tuples)"""
        for obs, action, next_obs, reward in play_data:
            if self.is_primitive(action):
                self.update_model(obs, action, next_obs, reward)

    def is_primitive(self, action) -> bool:
        raise NotImplementedError

    def update_model(self, obs, action, next_obs, reward):
        """Update the environment model *in-place* (deviation from article)"""
        self.n_transitions[obs, action, next_obs] += 1
        self.n_action_pairs[obs, action] += 1
        self.reward_action_pairs[obs, action] += reward

        # TODO: can't we do these operations at the end of learning only instead of at each iteration?
        if self.is_primitive(action):
            self.transition_model[obs, action][next_obs] = self.n_transitions[obs, action, next_obs] / self.n_action_pairs[obs, action]
            self.reward_model[obs, action] = self.reward_action_pairs[obs, action] / self.n_action_pairs[obs, action]
        
        else:
            # same as above???
            self.transition_model[obs, action][next_obs] = self.n_transitions[obs, action, next_obs] / self.n_action_pairs[obs, action]
            subtask = ...
            next_observations = ...
            optimal_subtask_action = self.act_subtask(obs, subtask)
            self.reward_model[obs, action] = self.reward[subtask, optimal_subtask_action] + sum(self.transition_model[obs, optimal_subtask_action, (subtask, _next_obs)] * self.reward_model[subtask, self.act_subtask(_next_obs, subtask)] for _next_obs in next_observations)

    def act_on_subtask(self, obs, subtask: int):
        raise NotImplementedError
    
    def load(self, folder_path: str):
        ...

    def save(self, folder_path: str):
        ...
