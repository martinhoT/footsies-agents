from .agent import GameModelAgent


def get_loggables(agent: GameModelAgent):
    return {
        "network_histograms": [
            *[g[1].network for g in agent.game_models],
        ],
        "custom_evaluators": [
            ("Learning/Loss_Guard", agent.evaluate_average_loss_guard),
            ("Learning/Loss_Move", agent.evaluate_average_loss_move),
            ("Learning/Loss_MoveProgress", agent.evaluate_average_loss_move_progress),
            ("Learning/Loss_Position", agent.evaluate_average_loss_position),
            ("Learning/Loss", agent.evaluate_average_loss_and_clear), # the evaluate loss one should be at the end since it's the one that clears the denominator counter
        ],
        "custom_evaluators_over_test_states": [],
    }
