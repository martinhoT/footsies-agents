import torch
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


class Observer:
    def update(self, step: int, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict, agent: FootsiesAgentBase | BaseAlgorithm):
        pass

    @property
    def data(self) -> tuple[list[int], tuple[list[float], ...]]:
        raise NotImplementedError("this observer didn't implement the 'data' property")

    @staticmethod
    def attributes() -> tuple[str, ...]:
        raise NotImplementedError("this observer didn't implement the 'attributes' property")


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

        # Never stop early
        return True


class WinRateObserver(Observer):
    def __init__(self, log_frequency: int = 1000, last: int = 100):
        self._log_frequency = log_frequency
        self._wins = deque([], maxlen=last)
        self._idxs = []
        self._values = []

    def update(self, step: int, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict, agent: FootsiesAgentBase):
        if terminated or truncated:
            won = (reward > 0) if terminated else (next_info["guard"][0] > next_info["guard"][1])
            self._wins.append(won)

        if step % self._log_frequency == 0:
            self._idxs.append(step)
            win_rate = sum(self._wins) / len(self._wins) if self._wins else 0.5
            self._values.append(win_rate)

    @property
    def data(self) -> tuple[list[int], tuple[list[float], ...]]:
        return self._idxs, (self._values,)

    @staticmethod
    def attributes() -> tuple[str, ...]:
        return ("win_rate",)


class GameModelObserver(Observer):
    def __init__(self, log_frequency: int = 1000):
        self._log_frequency = log_frequency
        self._idxs = []
        self._values = []

    def update(self, step: int, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict, agent: GameModelAgent):
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

    def update(self, step: int, obs: torch.Tensor, next_obs: torch.Tensor, reward: float, terminated: bool, truncated: bool, info: dict, next_info: dict, agent: MimicAgent):
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



T = TypeVar("T", bound=Observer)

def custom_loop(
    agent: FootsiesAgentBase | BaseAlgorithm | Callable[[Env], FootsiesAgentBase | BaseAlgorithm],
    label: str,
    id_: int,
    observer_type: type[T],
    opponent: Callable[[dict, dict], tuple[bool, bool, bool]] | None = None,
    initial_seed: int | None = 0,
    timesteps: int = 1000000,
) -> T:

    port_start = 11000 + 1000 * id_
    port_stop = 11000 + (1000) * (id_ + 1)
    env, footsies_env = create_eval_env(port_start=port_start, port_stop=port_stop)

    footsies_env.set_opponent(opponent)

    observer = observer_type()

    if isinstance(agent, Callable):
        agent = agent(env)

    if isinstance(agent, FootsiesAgentBase):
        return custom_loop_footsies(
            agent=agent,
            env=env,
            label=label,
            id_=id_,
            observer=observer,
            initial_seed=initial_seed,
            timesteps=timesteps,
        )
    
    else:
        return custom_loop_sb3(
            agent=agent,
            observer=observer,
            timesteps=timesteps
        )

def custom_loop_footsies(
    agent: FootsiesAgentBase,
    env: Env,
    label: str,
    id_: int,
    observer: T,
    initial_seed: int | None = 0,
    timesteps: int = int(1e6),
) -> T:

    process_id: int = mp.current_process()._identity[0] - 1

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

        obs, info = next_obs, next_info
    
    return observer


def custom_loop_sb3(
    agent: BaseAlgorithm,
    observer: T,
    timesteps: int = int(1e6),
) -> T:
    
    agent.learn(
        total_timesteps=timesteps,
        callback=ObserverSB3Callback(observer),
        log_interval=None, # type: ignore
    )

    return observer


def dataset_run(
    agent: MimicAgent | GameModelAgent,
    label: str,
    observer_type: type[T],
    seed: int | None = 0,
    epochs: int = 100,
    shuffle: bool = True,
) -> T:
    if not shuffle and seed is not None:
        raise ValueError("using a seed without shuffling is inconsequential, if no shuffling is to be performed please set the seed to `None`")

    # This is needed in case we are going to shuffle the dataset
    torch.manual_seed(seed)

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
        