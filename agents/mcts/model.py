class EnvironmentModel:
    def __init__(self):
        self.opponent_predictor
        self.game_model

    def incorporate(self, state, p1_action, p2_action, next_state):
        self.opponent_predictor.update(state, p2_action)
        self.game_model.update(state, p1_action, p2_action, next_state)

    def predict(self, state, p1_action) -> any:
        p2_action = self.opponent_predictor.predict(state)
        self.game_model.next_state(state, p1_action, p2_action)