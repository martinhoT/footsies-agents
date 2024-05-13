import importlib
import optuna
import os
from typing import Callable, cast
from gymnasium import Env
from agents.base import FootsiesAgentBase
from footsies_gym.envs.footsies import FootsiesEnv
from args import parse_args_experiment, ExperimentArgs
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.monitor import Monitor
from main import create_env, save_agent, setup_logger
from opponents.curriculum import CurriculumManager
from main import extract_opponent_manager
from tqdm import trange


DefineModelFunction = Callable[[optuna.Trial, Env], FootsiesAgentBase | BaseAlgorithm]
ObjectiveFunction = Callable[[FootsiesAgentBase | BaseAlgorithm, Env], float]

class TrialManager:
    def __init__(
        self,
        agent_name: str,
        env: Env,
        define_model_function: DefineModelFunction,
        objective_function: ObjectiveFunction | None,
        time_steps: int = 1000000,
        time_steps_before_eval: int = 5000,
        curriculum: CurriculumManager | None = None,
        pruning: bool = True,
    ):
        if objective_function is None and curriculum is None:
            raise ValueError("if not using the curriculum, a custom objective function has to be passed")
        if curriculum is not None and curriculum.episode_threshold is None:
            raise ValueError("the curriculum should have an episode threshold set, or else the objective cannot be properly calculated")

        self.agent_name = agent_name
        self.env = env
        self.footsies_env: FootsiesEnv = cast(FootsiesEnv, env.unwrapped)
        self.define_model = staticmethod(define_model_function)
        self.objective = staticmethod(objective_function) if objective_function is not None else self.curriculum_objective
        self.time_steps = time_steps
        self.time_steps_before_eval = time_steps_before_eval
        
        # Curriculum
        self._curriculum_episode_threshold: int = curriculum.episode_threshold if curriculum is not None and curriculum.episode_threshold is not None else 0
        self.curriculum = curriculum
        self.pruning = pruning

    def run(self, trial: optuna.Trial) -> float:
        if self.curriculum is not None:
            self.curriculum.reset()

        agent = self.define_model(trial, self.env)
        if isinstance(agent, FootsiesAgentBase):
            agent.preprocess(self.env)

        for step in trange(0, self.time_steps, self.time_steps_before_eval, position=0, colour="#ff4dde"):
            if isinstance(agent, BaseAlgorithm):
                agent.learn(self.time_steps_before_eval, progress_bar=True)
            else:
                terminated, truncated = True, True
                for _ in trange(self.time_steps_before_eval, position=1, colour="#4dffbe"):
                    if terminated or truncated:
                        obs, info = self.env.reset()
                        terminated, truncated = False, False
                    
                    action = agent.act(obs, info) # type: ignore
                    next_obs, reward, terminated, truncated, next_info = self.env.step(action)
                    
                    agent.update(obs, next_obs, reward, terminated, truncated, info, next_info) # type: ignore

            if self.pruning:
                loss = self.objective(agent, self.env)

                trial.report(loss, step + self.time_steps_before_eval)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            if self.curriculum is not None and self.curriculum.exhausted:
                break
    
        print(f"Finished trial {trial.number}")
        self.save(agent, trial.number)

        return self.objective(agent, self.env)

    def save(self, agent: FootsiesAgentBase | BaseAlgorithm, number: int):
        folder = os.path.join("scripts", "tuning", self.agent_name)
        name = f"tuning_{number}"
        save_agent(agent, name=name, folder=folder)
    
    def _curriculum_objective_compute(self, opponent_idx: int, episodes_taken: int, episode_threshold: int) -> float:
        significance = 10 ** opponent_idx
        return significance * (1.0 - episodes_taken / episode_threshold)

    def curriculum_objective(self, agent: FootsiesAgentBase | BaseAlgorithm, env: Env) -> float:
        if self.curriculum is None:
            raise RuntimeError("this method cannot be called if the curriculum is not set")
        
        past_objectives = sum(self._curriculum_objective_compute(
            opponent_idx=i,
            episodes_taken=self.curriculum.episodes_taken_for_opponent(i),
            episode_threshold=self._curriculum_episode_threshold,
        ) for i in range(self.curriculum.current_opponent_idx))

        # Current opponent assessment is incomplete, so we compute it separately
        current_opponent_objective = self._curriculum_objective_compute(
            opponent_idx=self.curriculum.current_opponent_idx,
            episodes_taken=self.curriculum.current_opponent_episodes,
            episode_threshold=self._curriculum_episode_threshold,
        )

        return past_objectives + current_opponent_objective


# NOTE: limitation, the different processes running this script should not be finding ports for FOOTSIES at the same time. As such, the processes should start up sequentially
def main(args: ExperimentArgs):
    if args.env.self_play.enabled:
        raise ValueError("self-play is not supported for tuning")

    if args.env.footsies_wrapper_simple.allow_agent_special_moves:
        print("WARNING: special moves were set to be allowed, but will be forcefully disabled for simplification")
        args.env.footsies_wrapper_simple.allow_agent_special_moves = False

    module_name = args.agent.name.replace(".", "_")

    optimize_module_str = ".".join(("scripts", "tuning", module_name))
    optimize_module = importlib.import_module(optimize_module_str)
    
    setup_logger(module_name, log_to_file=False, multiprocessing=False)
    
    define_model_function: DefineModelFunction = optimize_module.define_model
    # If we are training using the curriculum, then our objective is completely different
    objective_function: ObjectiveFunction | None
    if args.env.curriculum.enabled and args.curriculum_objective:
        objective_function = None
    else:
        objective_function = optimize_module.objective

    # Override curriculum arguments
    args.env.curriculum.episode_threshold = 1000
    args.env.curriculum.win_rate_threshold = 0.8
    args.env.curriculum.win_rate_over_episodes = 20
    
    ports = FootsiesEnv.find_ports(start=20000)
    args.env.kwargs.setdefault("game_port", ports["game_port"])
    args.env.kwargs.setdefault("opponent_port", ports["opponent_port"])
    args.env.kwargs.setdefault("remote_control_port", ports["remote_control_port"])
    
    args.env.kwargs.setdefault("game_path", "../Footsies-Gym/Build/FOOTSIES.x86_64")

    env = create_env(args.env, log_dir=None)

    curriculum = None
    if args.env.curriculum.enabled:
        curriculum = extract_opponent_manager(env)
        if curriculum is None:
            raise RuntimeError("the environment was not created with an opponent manager when one was requested, or it could not be found")
        elif not isinstance(curriculum, CurriculumManager):
            raise RuntimeError("the environment's opponent manager is not the curriculum one")

    # Wrap with a Monitor wrapper if using an SB3 agent, or else they may complain on evaluation
    if args.agent.is_sb3:
        env = Monitor(env)

    trialManager = TrialManager(
        agent_name=module_name + ("_curriculum" if args.env.curriculum.enabled else ""),
        env=env,
        define_model_function=define_model_function,
        objective_function=objective_function,
        time_steps=args.time_steps,
        time_steps_before_eval=args.time_steps_before_eval,
        curriculum=curriculum,
        pruning=args.pruning,
    )

    study_name = args.study_name if args.study_name is not None else module_name
    study_path = args.study_name if args.study_name is not None else os.path.join("scripts", "tuning", module_name)
    study = optuna.create_study(
        storage=f"sqlite:///{study_path}{'_curriculum' if args.env.curriculum.enabled else ''}.db",
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
