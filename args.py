import logging
import tyro
from dataclasses import dataclass, field
from intrinsic.base import IntrinsicRewardScheme
from intrinsic.counts import CountBasedScheme
from intrinsic.icm import ICMScheme
from intrinsic.rnd import RNDScheme
from typing import Literal, Dict, Annotated
from torch import nn


@dataclass
class MainArgs:
    misc: "MiscArgs"
    """Miscellaneous arguments"""
    agent: "AgentArgs"
    """Agent arguments"""
    env: "EnvArgs"
    """Environment arguments"""
    self_play: "SelfPlayArgs"
    """Self-play arguments"""

    episodes: int | None = None
    """Number of episodes. Doesn't work if an SB3 agent is used"""
    time_steps: int | None = None
    """Number of time steps"""
    penalize_truncation: float | None = None
    """How much to penalize the agent in case the environment is truncated, useful when a time limit is defined for instance. No penalization by default"""
    curriculum: bool = False
    """Perform curriculum learning on FOOTSIES with pre-made rule-based opponents"""
    intrinsic_reward_scheme_: Literal["count", "icm", "rnd", None] = None
    """The type of intrinsic reward to use (string specification)"""
    skip_freeze: bool = True
    """Skip any environment freeze in which an environment transition has equal observations. This is useful for handling hitstop in FOOTSIES"""
    seed: int | None = None
    """Random seed for both the agent and the environment. For the environment, it's only set on the first `reset()` call"""

    def __post_init__(self):
        if self.agent.is_sb3 and self.episodes is not None:
            raise ValueError("specifying a number of episodes for SB3 algorithms is not supported")

        if self.self_play.enabled and self.curriculum:
            raise ValueError("can't use both self-play and curriculum learning at the same time")
        
        if self.self_play.enabled or self.curriculum:
            self.env.kwargs["opponent"] = lambda o: (False, False, False)
        
        if self.env.is_footsies:
            if "game_path" not in self.env.kwargs:
                raise ValueError(
                    "the path to the FOOTSIES executable should be specified when using the FOOTSIES environment, through the environment keyword argument ('game_path')"
                )
        
        self.env.log_dir = f"{self.env.log_dir}/{self.agent.name}"

        if self.skip_freeze and not self.env.torch:
            raise ValueError("skipping environment freezes is not supported on observations that aren't PyTorch tensors")
    
    @property
    def intrinsic_reward_scheme(self) -> IntrinsicRewardScheme | None:
        """The type of intrinsic reward to use"""
        if self.intrinsic_reward_scheme_ == "count":
            return CountBasedScheme
        elif self.intrinsic_reward_scheme_ == "icm":
            return ICMScheme
        elif self.intrinsic_reward_scheme_ == "rnd":
            return RNDScheme

        return None


@dataclass
class MiscArgs:
    save: bool = True
    """Whether the agent is saved to disk after training"""
    load: bool = True
    """Whether the agent is loaded from disk before training"""
    
    log: bool = True
    """Whether the agent is logged (both standard and Tensorboard logs)"""
    log_frequency: int = 5000
    """Number of time steps between each Tensorboard log"""
    log_test_states_number: int = 5000
    """Number of test states to use when evaluating some metrics for Tensorboard logging"""
    log_step_start: int = 0
    """Value at which the logging time step will start, useful for appending to existing Tensorboard logs"""
    log_episode_start: int = 0
    """Value at which the logging episode will start, useful for appending to existing Tensorboard logs"""
    log_file_level_: Annotated[Literal["critical", "error", "warning", "info", "debug"], tyro.conf.arg(name="log_file_level")] = "debug"
    """The log level of the logs created in the log file. Recommended to be either `debug` or `info`"""
    log_stdout_level_: Annotated[Literal["critical", "error", "warning", "info", "debug"], tyro.conf.arg(name="log_stdout_level")] = "info"
    """The log level of the logs printed to the standard output. Recommended to be `info`"""

    hogwild: bool = False
    """Whether to use the Hogwild! asynchronous training algorithm. Only available for FOOTSIES agents based on PyTorch (for sharing of model parameters)"""
    hogwild_cpus: int | None = None
    """Maximum number of CPUs to use for Hogwild! training. If `None`, will use all available"""
    hogwild_n_workers: int = 6
    """Number of parallel workers to use for Hogwild! training"""

    @property
    def log_file_level(self) -> int:
        """The log level of the logs created in the log file"""
        return getattr(logging, self.log_file_level_.upper())

    @property
    def log_stdout_level(self) -> int:
        """The log level of the logs printed to the standard output"""
        return getattr(logging, self.log_stdout_level_.upper())


@dataclass
class AgentArgs:
    model: tyro.conf.Positional[str]
    """Name of the agent initializer/class to use, either from the 'models' folder or from Stable-Baselines3"""
    kwargs: tyro.conf.UseAppendAction[Dict[str, int | float | bool | str]] = field(default_factory=dict)
    """Keyword arguments to pass to the agent initializer/constructor"""
    name: str | None = None
    """The name of the agent, for saving, loading and logging. If `None`, will use `model` as the name"""
    is_sb3: bool = False
    """Whether the agent is a Stable-Baselines3 agent"""

    def __post_init__(self):
        self.name = self.name if self.name is not None else self.model
        self.is_sb3 = "sb3." == self.model[:4]
        if self.is_sb3:
            self.model = self.model[4:]


@dataclass
class EnvArgs:
    diayn: "DIAYNArgs"
    """Arguments for the DIAYN wrapper"""

    name: str = "FOOTSIES"
    """Gymnasium environment to use. The special value 'FOOTSIES' instantiates the FOOTSIES environment"""
    kwargs: tyro.conf.UseAppendAction[Dict[str, int | float | bool | str]] = field(default_factory=dict)
    """Keyword arguments to pass to the environment contructor"""
    wrapper_time_limit: int = 99 * 60 # NOTE: not actually sure if it's 60, for FOOTSIES it may be 50
    """Add a time limit wrapper to the environment, with the time limit being enforced after the given number of time steps. Defaults to a number equivalent to 99 seconds in FOOTSIES"""
    
    footsies_wrapper_norm: bool = True
    """Use the Normalized wrapper for FOOTSIES"""
    footsies_wrapper_norm_guard: bool = True
    """For the Normalized wrapper, whether to normalize the guard variable"""
    footsies_wrapper_acd: bool = False
    """Use the Action Combinations Discretized wrapper for FOOTSIES"""
    footsies_wrapper_simple: tuple[bool, bool, str, str] = (True, False, "last", "last")
    """Use the Simple Actions wrapper for FOOTSIES (tuple of `(enabled, remove_agent_special_moves, assumed_agent_action_on_nonactionable, assumed_opponent_action_on_nonactionable)`)"""
    footsies_wrapper_fs: bool = False
    """Use the Frame Skipped wrapper for FOOTSIES"""
    footsies_wrapper_adv: bool = False
    """Use the Encourage Advance wrapper for FOOTSIES"""
    footsies_wrapper_phasic: bool = False
    """Use the Phasic Move Progress wrapper for FOOTSIES"""
    footsies_wrapper_history: Dict[str, int | float | bool | str] | None = None
    """Use the Simple History wrapper for FOOTSIES, if not `None`. This specifies the arguments that should be passed to the wrapper (`p{1,2}`, `p{1,2}_n` and `p{1,2}_distinct`)"""
    
    torch: bool = True
    """Whether to transform environment observations to torch tensors"""
    log_dir: str = "runs"
    """The directory where any environment-related Tensorboard logs are to be saved"""

    @property
    def is_footsies(self) -> bool:
        """Whether the environment is FOOTSIES"""
        return self.name == "FOOTSIES"


@dataclass
class DIAYNArgs:
    enabled: bool = False
    """Use the DIAYN wrapper, which replaces the environment's reward with the pseudo-reward from DIAYN, fostering the creation of a diverse set of task-agnostic skills"""
    skill_dim: int = 5
    """The dimensionality of the skill vectors for DIAYN, roughly equivalent to the desired number of skills to learn"""
    include_baseline: bool = True
    """Whether to include the baseline in the pseudo-reward of DIAYN. Excluding it encourages haste rather than staying alive"""
    discriminator_learning_rate: float = 1e-3
    """The learning rate for the discriminator network in DIAYN"""
    discriminator_hidden_layer_sizes: list[int] = field(default_factory=lambda: [32])
    """Hidden layer sizes for the discriminator network in DIAYN"""
    discriminator_hidden_layer_activation_: str = "ReLU"
    """Hidden layer activation for the discriminator network in DIAYN (string specification)"""

    def discriminator_hidden_layer_activation(self) -> nn.Module:
        """Hidden layer activation for the discriminator network in DIAYN"""
        return getattr(nn, self.discriminator_hidden_layer_activation_str)()


@dataclass
class SelfPlayArgs:
    enabled: bool = False
    """Whether to use self-play during training on FOOTSIES. It's recommended to use the time limit wrapper"""
    max_opponents: int = 10
    """Maximum number of opponents to hold at once in the opponent pool. Doesn't count the in-game bot"""
    snapshot_interval: int = 2000
    """The interval between snapshots of the current policy for the opponent pool, in number of episodes"""
    switch_interval: int = 100
    """The interval between opponent switched, in number of episodes"""
    mix_bot: int = 1
    """How many opponents will the in-game opponent count as, when sampling from the opponent pool. In-game bot won't be sampled if 0"""
    add_curriculum_opps: bool = False
    """Add all opponents that are used for curriculum learning into the self-play's opponent pool"""


@dataclass
class ExperimentArgs:
    agent: "AgentArgs"
    """Agent arguments"""
    episodes: int = 15000
    """Number of episodes"""
    study_name: str = "test-experiment"
    """Name of the study (should be the same among processes on the same study)"""
    maximize: bool = False
    """Whether to maximize or minimize the objective value"""
    n_trials: int = 10
    """The number of trials to attempt"""


def parse_args() -> MainArgs:
    return tyro.cli(MainArgs)


def parse_args_experiment() -> ExperimentArgs:
    return tyro.cli(ExperimentArgs)
