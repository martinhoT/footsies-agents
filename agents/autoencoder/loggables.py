from .agent import AutoencoderAgent


def get_loggables(agent: AutoencoderAgent):
    return {
        "network_histograms": [
            agent.autoencoder.decoder,
            agent.autoencoder.encoder,
        ],
        "custom_evaluators": [
            ("Learning/Loss", lambda: agent.evaluate_average_loss(clear=False)[0]),
            ("Learning/Loss_Seq", lambda: agent.evaluate_average_loss(clear=True)[1]),
        ],
        "custom_evaluators_over_test_states": [],
    }
