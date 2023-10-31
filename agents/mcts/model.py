"""
The dynamically learned model of the environment.
It takes a state-action pair as input and returns a state as output.

The model is learned based on linear dependencies.

Examples:

[G3 G3]                 [G3 G3]
[STAND STAND]   -[R]->  [FORWARD STAND]
[F0 F0]                 [F0 F0]
[P-2 P2]                [P-1.9 P2]

Let's generalize. In any guard value, any move, at any frame and at any position, inputting RIGHT will yield the above difference, that is:
{Any} -[R]-> {p1.move = p1.move + FORWARD - STAND, p1.position = p1.position + (-1.9) - (-2)}

If later we find:

[G3 G3]                     [G3 G3]
[STAND N_ATTACK]    -[R]->  [DAMAGE N_ATTACK]
[F0 F5]                     [F0 F6]
[P-2 P0]                    [P-2.1 P0]

Then we need to restrict the input of all known rules to account for this case where it doesn't generalize.
We generalize the difference itself.
{Any - }


Note: in the environment model, we also need to take the opponent's action into account! (using a predicted opponent policy for example).
This also allows fine-tuning on the desired reaction time of the agent (according to Hick's law for instance):
if the entropy of the opponent policy's action distribution is too high, then the reaction time should be high as well to match

Note: the actions in the model should ideally be macro-actions (such as special moves), or else we don't have the Markov property
"""

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