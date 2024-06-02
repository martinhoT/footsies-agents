import os
import random
import torch as T
import pandas as pd
import multiprocessing as mp
from collections import deque
from typing import Callable, TypeVar, Sequence
from agents.base import FootsiesAgentBase
from agents.action import ActionMap
from tqdm import trange, tqdm
from scripts.evaluation.utils import create_eval_env
from data import FootsiesDataset, FootsiesTorchDataset
from torch.utils.data import DataLoader
from agents.game_model.agent import GameModelAgent
from agents.mimic.agent import MimicAgent
from gymnasium import Env
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from abc import ABC, abstractmethod
from typing import Protocol
from footsies_gym.envs.footsies import FootsiesEnv
from dataclasses import dataclass
from args import EnvArgs
from copy import deepcopy
from warnings import warn
import logging


LOGGER = logging.getLogger("scripts.evaluation.custom_loop")


class Observer(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, step: int, obs: T.Tensor, next_obs: T.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict, agent: FootsiesAgentBase | BaseAlgorithm):
        pass

    @property
    def data(self) -> tuple[list[int], tuple[list[float], ...]]:
        raise NotImplementedError("this observer didn't implement the 'data' property")

    @staticmethod
    def attributes() -> tuple[str, ...]:
        raise NotImplementedError("this observer didn't implement the 'attributes' property")

    def enough(self) -> bool:
        return False

    def df(self, attr_suffix: str = "") -> pd.DataFrame:
        idxs, attribute_values = self.data

        data: dict[str, Sequence[int | float]] = {"Idx": idxs}

        for attribute, values in zip(self.attributes(), attribute_values):
            data[attribute + attr_suffix] = values

        return pd.DataFrame(data)


class ObserverSB3Callback(BaseCallback):
    def __init__(self, observer: Observer, verbose: int = 0):
        super().__init__(verbose)

        self.observer = observer
        
        # This is only used in the case we cannot determine all observations in a transition.
        self._stored_obs: T.Tensor | None = None
    
    def _on_step(self) -> bool:
        truncated = self.locals["infos"][0]["TimeLimit.truncated"]
        terminated = not truncated and bool(self.locals["dones"][0].item())
        new_obs: T.Tensor = self.locals["new_obs"]

        # This is the case for DQN, so we are going to miss the very first transition of each episode.
        # This differential treatment should not affect the results since we are only wrapping the win rate observer,
        # which only cares about the end of episodes, and the "step" is obtained correctly.
        if "obs_tensor" not in self.locals:
            obs = self._stored_obs
            self._stored_obs = new_obs
            if obs is None:
                return True

        else:
            obs = self.locals["obs_tensor"]

        self.observer.update(
            step=self.num_timesteps - 1,
            obs=obs,
            next_obs=new_obs,
            reward=self.locals["rewards"][0].item(),
            terminated=terminated,
            truncated=truncated,
            info={},
            next_info=self.locals["infos"][0],
            agent=self.model,
        )

        # We should continue while the observer says that we have not collected enough data
        return not self.observer.enough()


class WinRateObserver(Observer):
    def __init__(self, log_frequency: int = 1000, last: int = 100):
        self._log_frequency = log_frequency
        self._wins = deque([], maxlen=last)
        self._idxs = []
        self._values = []

    def update(self, step: int, obs: T.Tensor, next_obs: T.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict, agent: FootsiesAgentBase):
        if terminated or truncated:
            won = (reward > 0) if terminated else (next_info["guard"][0] > next_info["guard"][1])
            self._wins.append(won)

        if step % self._log_frequency == 0:
            self._idxs.append(step)
            self._values.append(self.win_rate)

    @property
    def data(self) -> tuple[list[int], tuple[list[float], ...]]:
        return self._idxs, (self._values,)

    @staticmethod
    def attributes() -> tuple[str, ...]:
        return ("win_rate",)
    
    @property
    def win_rate(self) -> float:
        return sum(self._wins) / len(self._wins) if self._wins else 0.5


class GameModelObserver(Observer):
    def __init__(self, log_frequency: int = 1000, validation_set: tuple[T.Tensor, ...] | None = None):
        self._log_frequency = log_frequency
        self._validation_set = validation_set
        self._idxs = []
        self._values = []

    def update(self, step: int, obs: T.Tensor, next_obs: T.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict, agent: GameModelAgent):
        if step % self._log_frequency == 0:
            self._idxs.append(step)
            loss_guard_val = 0
            loss_move_val = 0
            loss_move_progress_val = 0
            loss_position_val = 0
            loss_val = 0
            if self._validation_set is not None:
                guard_loss, move_loss, move_progress_loss, position_loss = 0, 0, 0, 0

                obs, next_obs, _, p1_action, p2_action, _ = self._validation_set
                for offset, game_model in agent.game_models:
                    if offset > 1:
                        obs = obs[:-(offset - 1), :]
                        p1_action = p1_action[:-(offset - 1), :]
                        p2_action = p2_action[:-(offset - 1), :]
                        next_obs = next_obs[(offset - 1):, :]
                    
                    gl, ml, mpl, pl = game_model.update(obs, p1_action, p2_action, next_obs, actually_update=False)
                    guard_loss += gl
                    move_loss += ml
                    move_progress_loss += mpl
                    position_loss += pl

                loss_guard_val = guard_loss
                loss_move_val = move_loss
                loss_move_progress_val = move_progress_loss
                loss_position_val = position_loss
                loss_val = guard_loss + move_loss + move_progress_loss + position_loss

            print(step, loss_guard_val, loss_move_val, loss_move_progress_val, loss_position_val, loss_val)

            self._values.append((
                agent.evaluate_average_loss_guard(),
                agent.evaluate_average_loss_move(),
                agent.evaluate_average_loss_move_progress(),
                agent.evaluate_average_loss_position(),
                agent.evaluate_average_loss_and_clear(),
                loss_guard_val,
                loss_move_val,
                loss_move_progress_val,
                loss_position_val,
                loss_val,
            ))

    @property
    def data(self) -> tuple[list[int], tuple[list[float], ...]]:
        return self._idxs, tuple(zip(*self._values))

    @staticmethod
    def attributes() -> tuple[str, ...]:
        return ("loss_guard", "loss_move", "loss_move_progress", "loss_position", "loss", "loss_guard_val", "loss_move_val", "loss_move_progress_val", "loss_position_val", "loss_val")


class MimicObserver(Observer):
    def __init__(self, log_frequency: int = 1000, validation_set: tuple[T.Tensor, ...] | None = None):
        self._log_frequency = log_frequency
        self._validation_set = validation_set
        self._idxs = []
        self._values = []

        # Determine the points at which to reset the context
        self._p1_should_reset_context_at: list[int] | None = None
        self._p2_should_reset_context_at: list[int] | None = None

    def _compute_reset_context_at(self, agent: MimicAgent):
        if self._validation_set is None:
            raise ValueError("cannot compute where to reset the context because no validation set was provided")
        
        obs, next_obs, _, _, _, terminated = self._validation_set
        terminated = terminated
        
        self._p1_should_reset_context_at = []
        self._p2_should_reset_context_at = []

        for i, (o, no, t) in enumerate(zip(obs, next_obs, terminated)):
            # They are not batch-like...
            o = o.unsqueeze(0)
            no = no.unsqueeze(0)
            t = bool(t.item()) # convert to bool to satisfy Pylance
            
            if agent.p1_model is not None:
                if agent.p1_model.should_reset_context(o, no, t):
                    self._p1_should_reset_context_at.append(i)
            
            if agent.p2_model is not None:
                if agent.p2_model.should_reset_context(o, no, t):
                    self._p2_should_reset_context_at.append(i)
        
        # Add the final point (this also allows not resetting context at all to work).
        self._p1_should_reset_context_at.append(i + 1)
        self._p2_should_reset_context_at.append(i + 1)

        # The arrays should contain context lengths, not indices.
        for i in reversed(range(1, len(self._p1_should_reset_context_at))):
            self._p1_should_reset_context_at[i] -= self._p1_should_reset_context_at[i - 1]
        
        for i in reversed(range(1, len(self._p2_should_reset_context_at))):
            self._p2_should_reset_context_at[i] -= self._p2_should_reset_context_at[i - 1]

    def update(self, step: int, obs: T.Tensor, next_obs: T.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict, agent: MimicAgent):
        if step % self._log_frequency == 0:
            self._idxs.append(step)
            p1_loss_val = 0
            p2_loss_val = 0
            if self._validation_set is not None:
                loss_p1 = None
                loss_p2 = None

                obs, _, _, p1_action, p2_action, _ = self._validation_set

                # Determine the points at which to reset the context.
                # We assume it only needs to be done once since the agent should be the same.
                if self._p1_should_reset_context_at is None or self._p2_should_reset_context_at is None:
                    self._compute_reset_context_at(agent)

                assert self._p1_should_reset_context_at is not None
                assert self._p2_should_reset_context_at is not None

                obs_split_p1 = T.split(obs, self._p1_should_reset_context_at)
                p1_action_split = T.split(p1_action, self._p1_should_reset_context_at)

                obs_split_p2 = T.split(obs, self._p2_should_reset_context_at)
                p2_action_split = T.split(p2_action, self._p2_should_reset_context_at)

                loss_p1 = 0
                loss_p1_n = 0
                loss_p2 = 0
                loss_p2_n = 0

                for o, p1 in zip(obs_split_p1, p1_action_split):
                    if agent.p1_model is not None:
                        loss_p1 += agent.p1_model.compute_loss(o, p1)
                        loss_p1_n += 1
                for o, p2 in zip(obs_split_p2, p2_action_split):
                    if agent.p2_model is not None:
                        loss_p2 += agent.p2_model.compute_loss(o, p2)
                        loss_p2_n += 1

                p1_loss_val = loss_p1 / loss_p1_n
                p2_loss_val = loss_p2 / loss_p2_n
                
            self._values.append((
                agent.evaluate_p1_average_loss_and_clear(),
                agent.evaluate_p2_average_loss_and_clear(),
                p1_loss_val,
                p2_loss_val,
            ))

    @property
    def data(self) -> tuple[list[int], tuple[list[float], ...]]:
        return self._idxs, tuple(zip(*self._values))

    @staticmethod
    def attributes() -> tuple[str, ...]:
        return ("p1_loss", "p2_loss", "p1_loss_val", "p2_loss_val")


class PreCustomLoop(Protocol):
    def __call__(self, agent: FootsiesAgentBase | BaseAlgorithm, env: Env, footsies_env: FootsiesEnv, seed: int | None) -> None:
        ...

AgentCustom = FootsiesAgentBase | BaseAlgorithm | Callable[[Env], FootsiesAgentBase | BaseAlgorithm]

@dataclass
class AgentCustomRun:
    agent:          AgentCustom
    opponent:       Callable[[dict, dict], tuple[bool, bool, bool]] | None
    env_args:       EnvArgs | None = None
    pre_loop:       PreCustomLoop | None = None
    skip_freeze:    bool = True
    """This is a loop that is run before the main one, but on which data won't be collected"""


O = TypeVar("O", bound=Observer)

def custom_loop(
    run: AgentCustomRun,
    label: str,
    id_: int,
    observer_type: type[O],
    seed: int | None = 0,
    timesteps: int = int(1e6),
) -> tuple[O, FootsiesAgentBase | BaseAlgorithm]:

    port_start = 11000 + 25 * id_
    port_stop = 11000 + 25 * (id_ + 1)
    env, footsies_env = create_eval_env(port_start=port_start, port_stop=port_stop, use_custom_opponent=True, env_args=run.env_args)

    if isinstance(run.agent, Callable):
        agent = run.agent(env)
    else:
        # The 'run.agent' will be shared between instances of the same run with different seeds!
        # We need to make a distinct copy here.
        agent = deepcopy(run.agent)

    if run.skip_freeze and isinstance(agent, BaseAlgorithm):
        warn(f"Requested to run '{label}' with an SB3 algorithm, but while skipping hitstop/blockstop freeze. Note that skipping freeze has no effect on SB3 algorithms")

    if run.pre_loop is not None:
        run.pre_loop(agent, env, footsies_env, seed)

    footsies_env.set_opponent(run.opponent)
    observer = observer_type()

    if isinstance(agent, FootsiesAgentBase):
        observer = custom_loop_footsies(
            agent=agent,
            env=env,
            label=label,
            id_=id_,
            observer=observer,
            initial_seed=seed,
            timesteps=timesteps,
        )
    
    else:
        observer = custom_loop_sb3(
            agent=agent,
            observer=observer,
            timesteps=timesteps
        )

    env.close()

    return observer, agent


def custom_loop_footsies(
    agent: FootsiesAgentBase,
    env: Env,
    label: str,
    id_: int,
    observer: O,
    initial_seed: int | None = 0,
    timesteps: int = int(1e6),
    skip_freeze: bool = True,
) -> O:

    process_id: int = mp.current_process()._identity[0] - 1

    T.set_num_threads(1)
    T.manual_seed(initial_seed)
    random.seed(initial_seed)

    seed = initial_seed
    terminated, truncated = True, True
    for step in trange(timesteps, desc=f"{label} ({id_})", unit="step", position=process_id, dynamic_ncols=True, colour="#f5426c"):
        if (terminated or truncated):
            obs, info = env.reset(seed=seed)
            terminated, truncated = False, False
            seed = None

        action = agent.act(obs, info)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        reward = float(reward)

        # Skip hitstop freeze
        if skip_freeze:
            while next_info["p1_hitstun"] and next_info["p2_hitstun"] and obs.isclose(next_obs).all().item():
                action = agent.act(next_obs, next_info)
                next_obs, r, terminated, truncated, next_info = env.step(action)
                r = float(r)
                reward += r

        agent.update(obs, next_obs, reward, terminated, truncated, info, next_info)
        observer.update(step, obs, next_obs, reward, terminated, truncated, info, next_info, agent)

        if observer.enough():
            break

        obs, info = next_obs, next_info
    
    return observer


def custom_loop_sb3(
    agent: BaseAlgorithm,
    observer: O,
    timesteps: int = int(1e6),
    seed: int = 0,
) -> O:
    
    agent.set_random_seed(seed)

    agent.learn(
        total_timesteps=timesteps,
        callback=ObserverSB3Callback(observer),
        log_interval=None, # type: ignore
        progress_bar=True,
    )

    return observer


# NOTE: shuffling is done at the level of episodes, not transitions
def dataset_run(
    agent: MimicAgent | GameModelAgent,
    label: str,
    observer_type: type[O],
    seed: int | None = 0,
    epochs: int = 100,
    shuffle: bool = True,
) -> O:
    
    # Instantiate a distinct copy of the agent, to avoid it being shared with other runs (e.g. a run with different seeds).
    agent = deepcopy(agent)

    if not shuffle and seed is not None:
        raise ValueError("using a seed without shuffling is inconsequential, if no shuffling is to be performed please set the seed to `None`")
    
    # Avoid CPU oversubscription. We only ever use 1 thread per process.
    # This is important since we don't need to interact with the game, and so the agent is free to learn as fast as it can.
    # It's better to manage this since this function is meant to be used with multiprocessing.
    T.set_num_threads(1)

    # This is needed in case we are going to shuffle the dataset
    T.manual_seed(seed)
    random.seed(seed)

    process_id: int = mp.current_process()._identity[0] - 1

    base_dataset = FootsiesDataset.load("footsies-dataset")
    base_dataset.shuffle() # this will internally use the seed set in random
    train_base_dataset, validation_base_dataset = base_dataset.generate_split(0.9)
    # Stupid: we need to convert after the split; if it's done before, the conversion will be lost
    train_base_dataset.convert_to_simple_actions("last")
    validation_base_dataset.convert_to_simple_actions("last")
    train_dataset = FootsiesTorchDataset(train_base_dataset)
    validation_dataset = FootsiesTorchDataset(validation_base_dataset)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # Lazy way to create a whole tensor batch
    validation_set: tuple[T.Tensor, ...] = next(iter(DataLoader(validation_dataset, batch_size=len(validation_dataset), shuffle=False)))
    validation_set = (validation_set[0].float(), validation_set[1].float(), validation_set[2].float(), validation_set[3].long(), validation_set[4].long(), validation_set[5].bool())
    observer = observer_type(log_frequency=10000, validation_set=validation_set)

    step = 0
    for epoch in range(epochs):
        for obs, next_obs, reward, p1_action, p2_action, terminated in tqdm(dataloader, desc=f"{label} ({epoch + 1}/{epochs})", unit="it", position=process_id, dynamic_ncols=True, colour="#42f593"):
            obs = obs.float()
            next_obs = next_obs.float()
            p1_action = p1_action.long()
            p2_action = p2_action.long()

            # Discard hitstop/freeze
            if (ActionMap.is_in_hitstop_torch(obs, True) or ActionMap.is_in_hitstop_torch(obs, False)) and obs.isclose(next_obs).all():
                continue
            
            if isinstance(agent, GameModelAgent):
                agent.update_with_simple_actions(obs, p1_action, p2_action, next_obs)
            else:
                agent.update_with_simple_actions(obs, p1_action, p2_action, terminated)

            step += 1

            observer.update(step, obs, next_obs, reward, terminated, False, {}, {}, agent)

    return observer
        