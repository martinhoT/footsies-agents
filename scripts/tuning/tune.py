import importlib
import optuna
import os
from typing import Callable
from gymnasium import Env
from agents.base import FootsiesAgentBase
from footsies_gym.envs.footsies import FootsiesEnv
from args import parse_args_experiment, ExperimentArgs
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from main import create_env, save_agent


DefineModelFunction = Callable[[optuna.Trial, Env], FootsiesAgentBase | BaseAlgorithm]
ObjectiveFunction = Callable[[FootsiesAgentBase | BaseAlgorithm, Env], float]

class TrialManager:
    def __init__(
        self,
        agent_name: str,
        env: Env,
        define_model_function: DefineModelFunction,
        objective_function: ObjectiveFunction,
        time_steps: int = 1000000,
        time_steps_before_eval: int = 5000,
    ):
        self.agent_name = agent_name
        self.env = env
        self.define_model = staticmethod(define_model_function)
        self.objective = staticmethod(objective_function)
        self.time_steps = time_steps
        self.time_steps_before_eval = time_steps_before_eval

    def run(self, trial: optuna.Trial) -> float:
        agent = self.define_model(trial, self.env)

        for step in range(0, self.time_steps, self.time_steps_before_eval):
            if isinstance(agent, BaseAlgorithm):
                agent.learn(self.time_steps_before_eval, progress_bar=True)
            else:
                terminated, truncated = True, True
                for _ in range(self.time_steps_before_eval):
                    if terminated or truncated:
                        obs, info = self.env.reset()
                        terminated, truncated = False, False
                    
                    action = agent.act(obs, info) # type: ignore
                    next_obs, reward, terminated, truncated, next_info = self.env.step(action)
                    
                    agent.update(obs, next_obs, reward, terminated, truncated, info, next_info) # type: ignore

            loss = self.objective(agent, self.env)

            trial.report(loss, step + self.time_steps_before_eval)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
        print(f"Finished trial {trial.number}")
        self.save(agent, trial.number)

        return self.objective(agent, self.env)

    def save(self, agent: FootsiesAgentBase | BaseAlgorithm, number: int):
        folder = os.path.join("scripts", "tuning", self.agent_name)
        name = f"tuning_{number}"
        save_agent(agent, name=name, folder=folder)


# NOTE: limitation, the different processes running this script should not be finding ports for FOOTSIES at the same time. As such, the processes should start up sequentially
def main(args: ExperimentArgs):
    module_name = args.agent.name.replace(".", "_")

    optimize_module_str = ".".join(("scripts", "tuning", module_name))
    optimize_module = importlib.import_module(optimize_module_str)
    
    define_model_function: DefineModelFunction = optimize_module.define_model
    objective_function: ObjectiveFunction = optimize_module.objective

    ports = FootsiesEnv.find_ports(start=20000)
    args.env.kwargs.setdefault("game_port", ports["game_port"])
    args.env.kwargs.setdefault("opponent_port", ports["opponent_port"])
    args.env.kwargs.setdefault("remote_control_port", ports["remote_control_port"])
    
    args.env.kwargs.setdefault("game_path", "../Footsies-Gym/Build/FOOTSIES.x86_64")

    env = create_env(args.env)

    # Wrap with a Monitor wrapper if using an SB3 agent, or else they may complain on evaluation
    env = Monitor(env)

    trialManager = TrialManager(
        agent_name=module_name,
        env=env,
        define_model_function=define_model_function,
        objective_function=objective_function,
        time_steps=args.time_steps,
        time_steps_before_eval=args.time_steps_before_eval,
    )

    study_name = args.study_name if args.study_name is not None else module_name
    study_path = args.study_name if args.study_name is not None else os.path.join("scripts", "tuning", module_name)
    study = optuna.create_study(
        storage=f"sqlite:///{study_path}.db",
        sampler=None,   # TPESampler
        pruner=None,    # MedianPruner
        study_name=study_name,
        direction="maximize" if args.maximize else "minimize",
        load_if_exists=True,
    )
    study.optimize(trialManager.run, n_trials=args.n_trials, show_progress_bar=True)

    env.close()

    # Copied straight from https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics:")
    print("  Number of finished trials:", len(study.trials))
    print("  Number of pruned trials:", len(pruned_trials))
    print("  Number of complete trials:", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Id:", trial.number)
    print("  Value:", trial.value)
    print("  Params:")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    main(parse_args_experiment())
