import random
import torch as T
import multiprocessing as mp
from collections import deque
from typing import Callable, TypeVar
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


class Observer(ABC):
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


class ObserverSB3Callback(BaseCallback):
    def __init__(self, observer: Observer, verbose: int = 0):
        super().__init__(verbose)

        self.observer = observer
    
    def _on_step(self) -> bool:
        truncated = self.locals["infos"][0]["TimeLimit.truncated"]
        terminated = not truncated and bool(self.locals["dones"].item())

        self.observer.update(
            step=self.num_timesteps,
            obs=self.locals["obs_tensor"],
            next_obs=self.locals["new_obs"],
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
    def __init__(self, log_frequency: int = 1000):
        self._log_frequency = log_frequency
        self._idxs = []
        self._values = []

    def update(self, step: int, obs: T.Tensor, next_obs: T.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict, agent: GameModelAgent):
        if step % self._log_frequency == 0:
            self._idxs.append(step)
            self._values.append((
                agent.evaluate_average_loss_guard(),
                agent.evaluate_average_loss_move(),
                agent.evaluate_average_loss_move_progress(),
                agent.evaluate_average_loss_position(),
                agent.evaluate_average_loss_and_clear(),
            ))

    @property
    def data(self) -> tuple[list[int], tuple[list[float], ...]]:
        return self._idxs, tuple(zip(*self._values))

    @staticmethod
    def attributes() -> tuple[str, ...]:
        return ("loss_guard", "loss_move", "loss_move_progress", "loss_position", "loss")


class MimicObserver(Observer):
    def __init__(self, log_frequency: int = 1000):
        self._log_frequency = log_frequency
        self._idxs = []
        self._values = []

    def update(self, step: int, obs: T.Tensor, next_obs: T.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict, agent: MimicAgent):
        if step % self._log_frequency == 0:
            self._idxs.append(step)
            self._values.append((
                agent.evaluate_p1_average_loss_and_clear(),
                agent.evaluate_p2_average_loss_and_clear(),
            ))

    @property
    def data(self) -> tuple[list[int], tuple[list[float], ...]]:
        return self._idxs, tuple(zip(*self._values))

    @staticmethod
    def attributes() -> tuple[str, ...]:
        return ("p1_loss", "p2_loss")


class PreCustomLoop(Protocol):
    def __call__(self, agent: FootsiesAgentBase | BaseAlgorithm, env: Env, footsies_env: FootsiesEnv, seed: int | None) -> None:
        ...


O = TypeVar("O", bound=Observer)

def custom_loop(
    agent: FootsiesAgentBase | BaseAlgorithm | Callable[[Env], FootsiesAgentBase | BaseAlgorithm],
    label: str,
    id_: int,
    observer_type: type[O],
    opponent: Callable[[dict, dict], tuple[bool, bool, bool]] | None = None,
    initial_seed: int | None = 0,
    pre_loop: PreCustomLoop | None = None, # this is a loop that is run before the main one, but on which data won't be collected
    timesteps: int = int(1e6),
) -> O:

    port_start = 11000 + 25 * id_
    port_stop = 11000 + 25 * (id_ + 1)
    env, footsies_env = create_eval_env(port_start=port_start, port_stop=port_stop)

    if isinstance(agent, Callable):
        agent = agent(env)

    if pre_loop is not None:
        pre_loop(agent, env, footsies_env, initial_seed)

    footsies_env.set_opponent(opponent)
    observer = observer_type()

    if isinstance(agent, FootsiesAgentBase):
        observer = custom_loop_footsies(
            agent=agent,
            env=env,
            label=label,
            id_=id_,
            observer=observer,
            initial_seed=initial_seed,
            timesteps=timesteps,
        )
    
    else:
        observer = custom_loop_sb3(
            agent=agent,
            observer=observer,
            timesteps=timesteps
        )

    env.close()

    return observer


def custom_loop_footsies(
    agent: FootsiesAgentBase,
    env: Env,
    label: str,
    id_: int,
    observer: O,
    initial_seed: int | None = 0,
    timesteps: int = int(1e6),
) -> O:

    process_id: int = mp.current_process()._identity[0] - 1

    T.manual_seed(initial_seed)
    random.seed(initial_seed)

    seed = initial_seed
    terminated, truncated = True, True
    for step in trange(timesteps, desc=f"{label} ({id_})", unit="step", position=process_id, dynamic_ncols=True, colour="#80e4ed"):
        if (terminated or truncated):
            obs, info = env.reset(seed=seed)
            terminated, truncated = False, False
            seed = None

        action = agent.act(obs, info)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        reward = float(reward)

        # Skip hitstop freeze
        in_hitstop = ActionMap.is_in_hitstop_ori(next_info, True) or ActionMap.is_in_hitstop_ori(next_info, False)
        while in_hitstop and obs.isclose(next_obs).all():
            action = agent.act(obs, info)
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
    )

    return observer


def dataset_run(
    agent: MimicAgent | GameModelAgent,
    label: str,
    observer_type: type[O],
    seed: int | None = 0,
    epochs: int = 100,
    shuffle: bool = True,
) -> O:
    if not shuffle and seed is not None:
        raise ValueError("using a seed without shuffling is inconsequential, if no shuffling is to be performed please set the seed to `None`")

    # This is needed in case we are going to shuffle the dataset
    T.manual_seed(seed)

    process_id: int = mp.current_process()._identity[0] - 1

    dataset = FootsiesDataset.load("footsies-dataset")
    dataset = FootsiesTorchDataset(dataset)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle)

    observer = observer_type()

    step = 0
    for epoch in range(epochs):
        for obs, next_obs, reward, p1_action, p2_action, terminated in tqdm(dataloader, desc=f"{label} ({epoch})", unit="it", position=process_id, dynamic_ncols=True, colour="#42f593"):
            obs = obs.float()
            next_obs = next_obs.float()

            # Discard hitstop/freeze
            if (ActionMap.is_in_hitstop_torch(obs, True) or ActionMap.is_in_hitstop_torch(obs, False)) and obs.isclose(next_obs).all():
                continue

            p1_action, p2_action = ActionMap.simples_from_transition_torch(obs, next_obs)
            
            if isinstance(agent, GameModelAgent):
                agent.update_with_simple_actions(obs, p1_action, p2_action, next_obs)
            else:
                agent.update_with_simple_actions(obs, p1_action, p2_action, terminated)

            step += 1

            observer.update(step, obs, next_obs, reward, terminated, False, {}, {}, agent)
    
    return observer
        