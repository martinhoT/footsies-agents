import os
import importlib
import optuna
import psutil
from itertools import count
from typing import Callable
from gymnasium import Env
from gymnasium.wrappers.flatten_observation import FlattenObservation
from agents.base import FootsiesAgentBase
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.normalization import FootsiesNormalized
from args import parse_args_experiment


def find_footsies_ports() -> tuple[int, int, int]:
    closed_ports = {p.laddr.port for p in psutil.net_connections(kind="tcp4")}

    ports = []

    for port in count(start=11000, step=1):
        if port not in closed_ports:
            ports.append(port)

        if len(ports) >= 3:
            break

    return tuple(ports)


class TrialManager:
    def __init__(
        self,
        env: Env,
        define_model_function: Callable[[optuna.Trial], FootsiesAgentBase],
        objective_function: Callable[[FootsiesAgentBase], float],
        total_episodes: int,
    ):
        self.env = env
        self.define_model = staticmethod(define_model_function)
        self.objective = staticmethod(objective_function)
        self.total_episodes = total_episodes

    def run(self, trial: optuna.Trial) -> float:
        agent = self.define_model(self.env.observation_space, self.env.action_space, trial)
        step = 0

        for _ in range(self.total_episodes):
            obs, info = self.env.reset()

            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = agent.act(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                
                agent.update(obs, reward, terminated, truncated, info)
                
                step += 1

            loss = self.objective(agent)

            trial.report(loss, step)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
        return self.objective(agent)


# NOTE: limitation, the different processes running this script should not be finding ports for FOOTSIES at the same time. As such, the processes should start up sequentially
if __name__ == "__main__":
    args = parse_args_experiment()

    optimize_module_str = ".".join(("agents", args.agent_name, "optimize"))
    optimize_module = importlib.import_module(optimize_module_str)
    
    define_model_function: Callable[[optuna.Trial], FootsiesAgentBase] = optimize_module.define_model
    objective_function: Callable[[FootsiesAgentBase], float] = optimize_module.objective

    game_port, opponent_port, remote_control_port = find_footsies_ports()

    env = FootsiesActionCombinationsDiscretized(
        FlattenObservation(
            FootsiesNormalized(
                FootsiesEnv(
                    game_path="../Footsies-Gym/Build/FOOTSIES.x86_64",
                    game_port=game_port,
                    opponent_port=opponent_port,
                    remote_control_port=remote_control_port,
                    
                    # TODO: should probably be specified through CLI arguments
                    by_example=True,
                )
            )
        )
    )

    trialManager = TrialManager(
        env=env,
        define_model_function=define_model_function,
        objective_function=objective_function,
        total_episodes=args.total_episodes,
    )

    study = optuna.create_study(
        storage=f"sqlite:///{args.study_name}.db",
        sampler=None,   # TPESampler
        pruner=None,    # MedianPruner
        study_name=args.study_name,
        direction=args.direction,
        load_if_exists=True,
    )
    study.optimize(trialManager.run, n_trials=args.n_trials, show_progress_bar=True)

    env.close()

    # Copied straight from https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
