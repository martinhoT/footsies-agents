from .agent import FootsiesAgent


def get_loggables(agent: FootsiesAgent):
    return {
        "network_histograms": [
            agent.representation_module,
            agent.game_model,
            agent.opponent_model,
            agent.actor_critic.actor,
            agent.actor_critic.critic,
        ],
        "custom_evaluators": [
            ("Learning/Delta", agent.evaluate_average_delta),
            ("Learning/Loss_Game_Model", agent.evaluate_average_loss_game_model),
            ("Learning/Loss_Opponent_Model", agent.evaluate_average_loss_opponent_model),
        ],
        "custom_evaluators_over_test_states": [],
    }
