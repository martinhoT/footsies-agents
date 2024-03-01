import torch
import argparse
import multiprocessing
import random
from dataclasses import dataclass
from rps.run import main
from functools import partial, reduce


@dataclass
class Configuration:
    self_play:                          bool
    dense_reward:                       bool
    health:                             int
    use_temporal_actions:               bool
    observation_include_move:           bool
    observation_include_move_progress:  bool
    opponent:                           str
    actor_lr:                           float
    critic_lr:                          float
    actor_entropy_loss_coef:            float
    actor_hidden_layer_sizes:           list[int]
    actor_hidden_layer_activation:      str
    critic_hidden_layer_sizes:          list[int]
    critic_hidden_layer_activation:     str
    time_limit:                         int
    time_limit_as_truncation:           bool

    def label(self) -> str:
        convert_bool = lambda value: "T" if value else "F"
        return f"{convert_bool(self.self_play)}_{convert_bool(self.dense_reward)}_{self.health}_{convert_bool(self.use_temporal_actions)}_{convert_bool(self.observation_include_move)}_{convert_bool(self.observation_include_move_progress)}_{self.opponent}_{self.actor_lr:.0e}_{self.critic_lr:.0e}_{self.actor_entropy_loss_coef:.2f}_{self.actor_hidden_layer_sizes}_{self.actor_hidden_layer_activation}_{self.critic_hidden_layer_sizes}_{self.critic_hidden_layer_activation}_{self.time_limit}_{convert_bool(self.time_limit_as_truncation)}"

    def markdown(self) -> str:
        """Returns a markdown representation of this configuration, for use in a table in the Logseq documentation"""
        convert_bool = lambda value: "{{:green-ball}}" if value else "{{:red-ball}}"
        return "| " + " | ".join([
            convert_bool(self.self_play),
            convert_bool(self.dense_reward),
            str(self.health),
            convert_bool(self.use_temporal_actions),
            convert_bool(self.observation_include_move),
            convert_bool(self.observation_include_move_progress),
            self.opponent,
            f"{self.actor_lr:.0e}",
            f"{self.critic_lr:.0e}",
            f"{self.actor_entropy_loss_coef:.2f}",
            str(self.actor_hidden_layer_sizes),
            self.actor_hidden_layer_activation,
            str(self.critic_hidden_layer_sizes),
            self.critic_hidden_layer_activation,
            str(self.time_limit),
            convert_bool(self.time_limit_as_truncation),
        ]) + " |"


ATTRIBUTE_VARIATIONS_NORMAL = {
    "self_play": [False, True],
    "dense_reward": [False, True],
    "health": [1, 3, 5],
    "use_temporal_actions": [False],
    "observation_include_move": [False, True],
    "observation_include_move_progress": [False],
    "opponent": ["uniform_random_play", "rocky_play"],
    "actor_lr": [1e-4],
    "critic_lr": [1e-4],
    "actor_entropy_loss_coef": [0.0, 0.05],
    "actor_hidden_layer_sizes": [[], [32]],
    "actor_hidden_layer_activation": ["ReLU"],
    "critic_hidden_layer_sizes": [[], [32]],
    "critic_hidden_layer_activation": ["ReLU"],
    "time_limit": [18], # (5 + (5 - 4)) * 2, function of healths (aka number of rounds), we consider the maximum health
    "time_limit_as_truncation": [False, True],
}


ATTRIBUTE_VARIATIONS_TEMPORAL = {
    "self_play": [False, True],
    "dense_reward": [False, True],
    "health": [1, 3, 5],
    "use_temporal_actions": [True],
    "observation_include_move": [False, True],
    "observation_include_move_progress": [False, True],
    "opponent": ["uniform_random_play_temporal", "masher", "dodge-attacker"],
    "actor_lr": [1e-4],
    "critic_lr": [1e-4],
    "actor_entropy_loss_coef": [0.0, 0.05],
    "actor_hidden_layer_sizes": [[], [32]],
    "actor_hidden_layer_activation": ["ReLU"],
    "critic_hidden_layer_sizes": [[], [32]],
    "critic_hidden_layer_activation": ["ReLU"],
    "time_limit": [18], # (5 + (5 - 4)) * 2, function of healths (aka number of rounds), we consider the maximum health
    "time_limit_as_truncation": [False, True],
}


def generate_configurations(variations: dict[str, list]) -> list[Configuration]:
    return [
        Configuration(
            self_play=self_play,
            dense_reward=dense_reward,
            health=health,
            use_temporal_actions=use_temporal_actions,
            observation_include_move=observation_include_move,
            observation_include_move_progress=observation_include_move_progress,
            opponent=opponent,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            actor_entropy_loss_coef=actor_entropy_loss_coef,
            actor_hidden_layer_sizes=actor_hidden_layer_sizes,
            actor_hidden_layer_activation=actor_hidden_layer_activation,
            critic_hidden_layer_sizes=critic_hidden_layer_sizes,
            critic_hidden_layer_activation=critic_hidden_layer_activation,
            time_limit=time_limit,
            time_limit_as_truncation=time_limit_as_truncation,
        )
        for self_play in variations["self_play"]
        for dense_reward in variations["dense_reward"]
        for health in variations["health"]
        for use_temporal_actions in variations["use_temporal_actions"]
        for observation_include_move in variations["observation_include_move"]
        for observation_include_move_progress in variations["observation_include_move_progress"]
        for opponent in variations["opponent"]
        for actor_lr in variations["actor_lr"]
        for critic_lr in variations["critic_lr"]
        for actor_entropy_loss_coef in variations["actor_entropy_loss_coef"]
        for actor_hidden_layer_sizes in variations["actor_hidden_layer_sizes"]
        for actor_hidden_layer_activation in variations["actor_hidden_layer_activation"]
        for critic_hidden_layer_sizes in variations["critic_hidden_layer_sizes"]
        for critic_hidden_layer_activation in variations["critic_hidden_layer_activation"]
        for time_limit in variations["time_limit"]
        for time_limit_as_truncation in variations["time_limit_as_truncation"]
    ]


def calculate_total_variations(variations: dict[str, list]) -> int:
    return reduce(int.__mul__, (len(v) for v in variations.values()))


CONFIGURATIONS: list[Configuration] = [
    *generate_configurations(ATTRIBUTE_VARIATIONS_NORMAL),
    *generate_configurations(ATTRIBUTE_VARIATIONS_TEMPORAL),
]


def run_configuration(configuration: Configuration, episodes: int = 10000) -> tuple[float, float]:
    plot_save_path = "rps/plots/" + configuration.label() + ".png"

    deltas, rewards, episode_lengths = main(
        episodes=episodes,
        actor_lr=configuration.actor_lr,
        critic_lr=configuration.critic_lr,
        actor_entropy_loss_coef=configuration.actor_entropy_loss_coef,
        actor_hidden_layer_sizes=configuration.actor_hidden_layer_sizes,
        actor_hidden_layer_activation=getattr(torch.nn, configuration.actor_hidden_layer_activation),
        critic_hidden_layer_sizes=configuration.critic_hidden_layer_sizes,
        critic_hidden_layer_activation=getattr(torch.nn, configuration.critic_hidden_layer_activation),
        self_play=configuration.self_play,
        dense_reward=configuration.dense_reward,
        health=configuration.health,
        use_temporal_actions=configuration.use_temporal_actions,
        observation_include_move=configuration.observation_include_move,
        observation_include_move_progress=configuration.observation_include_move_progress,
        opponent=configuration.opponent,
        time_limit=configuration.time_limit,
        time_limit_as_truncation=configuration.time_limit_as_truncation,
        plot=True,
        plot_during_training=False,
        plot_save_path=plot_save_path,
        interactive=False,
        log=True,
    )

    print(f"Configuration '{configuration.label()}' completed")

    # We only care about the last values, or else we will be accumulating tons of data (millions of episodes!)
    return deltas[-1], rewards[-1], episode_lengths[-1]


def main_multi(n_processes: int, episodes: int = 10000, configs: list[Configuration] = CONFIGURATIONS, md_out: str = None, part: int = None):
    print("Total configurations:", len(configs))

    # Divide the configurations into parts so that a run doesn't take forever
    if part is not None:
        # Shuffle the configurations so that configurations that take longer don't all sit in the same regions in the list
        shuffle_seed = 11037
        random.seed(shuffle_seed)
        random.shuffle(configs)
        # Get only a part of the configurations
        part_size = 100
        configs = configs[part * part_size:(part + 1) * part_size]

    ans = input(f"{len(configs)} configurations will be run with {n_processes} processes, proceed? (y/[n]) ")
    if ans.lower() != "y":
        print("Not accepted, exiting")
        return

    with multiprocessing.Pool(n_processes) as pool:
        results = pool.map(partial(run_configuration, episodes=episodes), configs)
    
    if md_out is None:
        print("Markdown table rows:")
        for configuration, (last_delta, last_reward, last_episode_length) in zip(configs, results):
            print(configuration.markdown() + f" {last_delta:.2e} | {last_reward:.2e} | {last_episode_length:.2f} |")

    else:
        print(f"Saving markdown table to '{md_out}'...", end=" ")
        with open(md_out, "at") as f:
            f.writelines((configuration.markdown() + f" {last_delta:.2e} | {last_reward:.2e} | {last_episode_length:.2f} |\n") for configuration, (last_delta, last_reward, last_episode_length) in zip(configs, results))
        print("done!")

def parse_args() -> dict[str, int]:
    parser = argparse.ArgumentParser("rps_multi", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n-processes", type=int, default=12, help="Number of parallel processes to use")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes to train for")
    parser.add_argument("--md-out", type=str, default=None, help="The path to save the markdown table to. If None, then the table will be printed to the console")
    parser.add_argument("--part", type=int, default=None, help="The part of the configurations to run. If not specified, then all configurations will be run")
    
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    main_multi(**parse_args())
