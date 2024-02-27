import torch
import argparse
import multiprocessing
from dataclasses import dataclass
from rps.run import main
from functools import partial, reduce


@dataclass
class Configuration:
    self_play:                          bool
    dense_reward:                       bool
    health:                             int
    use_temporal_actions:               bool
    observation_include_play:           bool
    observation_include_move_progress:  bool
    opponent:                           str
    actor_lr:                           float
    critic_lr:                          float
    actor_entropy_loss_coef:            float
    actor_hidden_layer_sizes:           list[int]
    actor_hidden_layer_activation:      str
    critic_hidden_layer_sizes:          list[int]
    critic_hidden_layer_activation:     str

    def label(self) -> str:
        convert_bool = lambda value: "T" if value else "F"
        return f"{convert_bool(self.self_play)}_{convert_bool(self.dense_reward)}_{self.health}_{convert_bool(self.use_temporal_actions)}_{convert_bool(self.observation_include_play)}_{convert_bool(self.observation_include_move_progress)}_{self.opponent}_{self.actor_lr:.0e}_{self.critic_lr:0e}_{self.actor_entropy_loss_coef:.2f}_{self.actor_hidden_layer_sizes}_{self.actor_hidden_layer_activation}_{self.critic_hidden_layer_sizes}_{self.critic_hidden_layer_activation}"

    def markdown(self) -> str:
        """Returns a markdown representation of this configuration, for use in a table in the Logseq documentation"""
        convert_bool = lambda value: "{{:green-ball}}" if value else "{{:red-ball}}"
        return "| " + " | ".join([
            convert_bool(self.self_play),
            convert_bool(self.dense_reward),
            str(self.health),
            convert_bool(self.use_temporal_actions),
            convert_bool(self.observation_include_play),
            convert_bool(self.observation_include_move_progress),
            self.opponent,
            f"{self.actor_lr:.0e}",
            f"{self.critic_lr:0e}",
            f"{self.actor_entropy_loss_coef:.2f}",
            str(self.actor_hidden_layer_sizes),
            self.actor_hidden_layer_activation,
            str(self.critic_hidden_layer_sizes),
            self.critic_hidden_layer_activation,
        ]) + " |"


ATTRIBUTE_VARIATIONS_NORMAL = {
    "self_play": [False, True],
    "dense_reward": [False, True],
    "health": [1, 3, 30],
    "use_temporal_actions": [False],
    "observation_include_play": [False, True],
    "observation_include_move_progress": [False],
    "opponent": ["uniform_random_play", "rocky_play"],
    "actor_lr": [1e-4, 1e-2],
    "critic_lr": [1e-4, 1e-2],
    "actor_entropy_loss_coef": [0.0, 0.05],
    "actor_hidden_layer_sizes": [[], [32]],
    "actor_hidden_layer_activation": ["ReLU"],
    "critic_hidden_layer_sizes": [[], [32]],
    "critic_hidden_layer_activation": ["ReLU"],
}


ATTRIBUTE_VARIATIONS_TEMPORAL = {
    "self_play": [False, True],
    "dense_reward": [False, True],
    "health": [1, 3, 30],
    "use_temporal_actions": [True],
    "observation_include_play": [False, True],
    "observation_include_move_progress": [False, True],
    "opponent": ["uniform_random_play_temporal", "masher", "dodge-attacker"],
    "actor_lr": [1e-4, 1e-2],
    "critic_lr": [1e-4, 1e-2],
    "actor_entropy_loss_coef": [0.0, 0.05],
    "actor_hidden_layer_sizes": [[], [32]],
    "actor_hidden_layer_activation": ["ReLU"],
    "critic_hidden_layer_sizes": [[], [32]],
    "critic_hidden_layer_activation": ["ReLU"],
}


def generate_configurations(variations: dict[str, list]) -> list[Configuration]:
    return [
        Configuration(
            self_play=self_play,
            dense_reward=dense_reward,
            health=health,
            use_temporal_actions=use_temporal_actions,
            observation_include_play=observation_include_play,
            observation_include_move_progress=observation_include_move_progress,
            opponent=opponent,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            actor_entropy_loss_coef=actor_entropy_loss_coef,
            actor_hidden_layer_sizes=actor_hidden_layer_sizes,
            actor_hidden_layer_activation=actor_hidden_layer_activation,
            critic_hidden_layer_sizes=critic_hidden_layer_sizes,
            critic_hidden_layer_activation=critic_hidden_layer_activation,
        )
        for self_play in variations["self_play"]
        for dense_reward in variations["dense_reward"]
        for health in variations["health"]
        for use_temporal_actions in variations["use_temporal_actions"]
        for observation_include_play in variations["observation_include_play"]
        for observation_include_move_progress in variations["observation_include_move_progress"]
        for opponent in variations["opponent"]
        for actor_lr in variations["actor_lr"]
        for critic_lr in variations["critic_lr"]
        for actor_entropy_loss_coef in variations["actor_entropy_loss_coef"]
        for actor_hidden_layer_sizes in variations["actor_hidden_layer_sizes"]
        for actor_hidden_layer_activation in variations["actor_hidden_layer_activation"]
        for critic_hidden_layer_sizes in variations["critic_hidden_layer_sizes"]
        for critic_hidden_layer_activation in variations["critic_hidden_layer_activation"]
    ]


def calculate_total_variations(variations: dict[str, list]) -> int:
    return reduce(int.__mul__, (len(v) for v in variations.values()))


CONFIGURATIONS: list[Configuration] = [
    *generate_configurations(ATTRIBUTE_VARIATIONS_NORMAL),
    *generate_configurations(ATTRIBUTE_VARIATIONS_TEMPORAL),
]


def run_configuration(configuration: Configuration, episodes: int = 10000) -> tuple[float, float]:
    plot_save_path = "rps/plots/" + configuration.label() + ".png"

    deltas, rewards = main(
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
        observation_include_play=configuration.observation_include_play,
        observation_include_move_progress=configuration.observation_include_move_progress,
        opponent=configuration.opponent,
        plot=True,
        plot_during_training=False,
        plot_save_path=plot_save_path,
        interactive=False,
        log=False,
    )

    print(f"Configuration '{configuration.label()}' completed")

    # We only care about the last values, or else we will be accumulating tons of data (millions of episodes!)
    return deltas[-1], rewards[-1]


def main_multi(n_processes: int, episodes: int = 10000, configs: list[Configuration] = CONFIGURATIONS, md_out: str = None):
    ans = input(f"{len(configs)} configurations will be run with {n_processes} processes, proceed? (y/[n]) ")
    if ans.lower() != "y":
        print("Not accepted, exiting")
        return

    with multiprocessing.Pool(n_processes) as pool:
        results = pool.map(partial(run_configuration, episodes=episodes), configs)
    
    if md_out is None:
        print("Markdown table rows:")
        for configuration, (last_delta, last_reward) in zip(configs, results):
            print(configuration.markdown() + f" {last_delta:.2e} | {last_reward:.2e} |")

    else:
        print(f"Saving markdown table to '{md_out}'...", end=" ")
        with open(md_out, "at") as f:
            f.writelines((configuration.markdown() + f" {last_delta:.2e} | {last_reward:.2e} |\n") for configuration, (last_delta, last_reward) in zip(configs, results))
        print("done!")

def parse_args() -> dict[str, int]:
    parser = argparse.ArgumentParser("rps_multi", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n-processes", type=int, default=12, help="Number of parallel processes to use")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes to train for")
    parser.add_argument("--md-out", type=str, default=None, help="The path to save the markdown table to. If None, then the table will be printed to the console")
    
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    main_multi(**parse_args())
