import tyro
from scripts.evaluation.s.accumulate_on_agent_frameskip import main as accumulate_on_agent_frameskip
from scripts.evaluation.s.action_masking import main as action_masking
from scripts.evaluation.s.actor_entropy_coef import main as actor_entropy_coef
from scripts.evaluation.s.adaptation_speed import main as adaptation_speed
from scripts.evaluation.s.advantage_formula import main as advantage_formula
from scripts.evaluation.s.assumed_opponent_action_on_frameskip import main as assumed_opponent_action_on_frameskip
from scripts.evaluation.s.baseline_compare import main as baseline_compare
from scripts.evaluation.s.consider_opponent_actions import main as consider_opponent_actions
from scripts.evaluation.s.critic_opponent_update_style import main as critic_opponent_update_style
from scripts.evaluation.s.discount_factor import main as discount_factor
from scripts.evaluation.s.game_model_method import main as game_model_method
from scripts.evaluation.s.hitstop_freeze import main as hitstop_freeze
from scripts.evaluation.s.opponent_model_dynamic_loss import main as opponent_model_dynamic_loss
from scripts.evaluation.s.opponent_model_entropy_coef import main as opponent_model_entropy_coef
from scripts.evaluation.s.opponent_model_recurrent import main as opponent_model_recurrent
from scripts.evaluation.s.reaction_time import main as reaction_time
from scripts.evaluation.s.self_play import main as self_play
from scripts.evaluation.s.sparse_vs_dense_reward_curriculum import main as sparse_vs_dense_reward_curriculum
from scripts.evaluation.s.sparse_vs_dense_reward import main as sparse_vs_dense_reward
from scripts.evaluation.s.special_moves import main as special_moves
from scripts.evaluation.s.target_network import main as target_network
from scripts.evaluation.s.transfer_learning import main as transfer_learning
from scripts.evaluation.s.zero_sum import main as zero_sum
from typing import Protocol
from traceback import print_exception
from sys import stdout


class EvaluationScript(Protocol):
    def __call__(self, *, seeds: int = 10, timesteps: int = int(1e6), processes: int = 12, y: bool = False) -> None:
        ...


EVALUATIONS: dict[str, EvaluationScript] = {
    "accumulate_on_agent_frameskip":        accumulate_on_agent_frameskip,
    "action_masking":                       action_masking,
    "actor_entropy_coef":                   actor_entropy_coef,
    "adaptation_speed":                     adaptation_speed,
    "advantage_formula":                    advantage_formula,
    "assumed_opponent_action_on_frameskip": assumed_opponent_action_on_frameskip,
    "critic_opponent_update_style":         critic_opponent_update_style,
    "consider_opponent_actions":            consider_opponent_actions,
    "discount_factor":                      discount_factor,
    "game_model_method":                    game_model_method,
    "hitstop_freeze":                       hitstop_freeze,
    "opponent_model_dynamic_loss":          opponent_model_dynamic_loss,
    "opponent_model_entropy_coef":          opponent_model_entropy_coef,
    "opponent_model_recurrent":             opponent_model_recurrent,
    "reaction_time":                        reaction_time,
    "sparse_vs_dense_reward_curriculum":    sparse_vs_dense_reward_curriculum,
    "sparse_vs_dense_reward":               sparse_vs_dense_reward,
    "special_moves":                        special_moves,
    "target_network":                       target_network,
    "transfer_learning":                    transfer_learning,
    "zero_sum":                             zero_sum,
    # Put self-play as the last one since it's the one that takes the longest to execute
    "self_play":                            self_play,
    # Actually put baseline comparison as the last one since it may change (if we perform more tuning)
    "baseline_compare":                     baseline_compare,
}


def main(seeds: int = 10, processes: int = 12, reverse: bool = False):
    evals = EVALUATIONS.items()
    if reverse:
        evals = reversed(evals)

    for name, script in evals:
        print("-" * 50)
        print(f"{'RUNNING ' + name:^50}")
        print("-" * 50)

        try:
            script(
                seeds=seeds,
                processes=processes,
                y=True,
            )
        
        except KeyboardInterrupt:
            print(f"Okay, runs of '{name}' were interrupted")
            return
        
        # Catch any problem and ignore it, just notify and try the next script
        except Exception as e:
            print("-" * 50)
            print(f"{'ERROR ' + name:^50}")
            print_exception(e, file=stdout)
            print("-" * 50)
            continue

        print("-" * 50)
        print(f"{'COMPLETED ' + name:^50}")
        print("-" * 50)


if __name__ == "__main__":
    tyro.cli(main)
