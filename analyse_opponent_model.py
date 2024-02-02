import dearpygui.dearpygui as dpg
import numpy as np
import torch
from gymnasium.wrappers.flatten_observation import FlattenObservation
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.normalization import FootsiesNormalized
from analysis import Analyser
from agents.mimic.agent import FootsiesAgent as OpponentModelAgent
from agents.utils import FOOTSIES_ACTION_MOVES, FOOTSIES_ACTION_MOVE_INDICES_MAP
from main import load_agent_model


# It's just a glorified dictionary
class ActionTableEstimator:
    def __init__(
        self,
        n_moves: int,
    ):
        self.n_moves = n_moves
        self.table = {}
    
    def update(self, observation: np.ndarray, move):
        obs = observation.tobytes()
        if obs not in self.table:
            self.table[obs] = np.zeros((self.n_moves,))
        
        self.table[obs][move] += 1
    
    def probability_distribution(self, observation) -> np.ndarray:
        obs = observation.tobytes()
        return self.table[obs] / np.sum(self.table[obs])

    def size(self) -> int:
        return len(self.table)


class OpponentDistributionPlot:
    def __init__(
        self,
        title: str,
        bar_width: int = 0.2,
    ):
        self.title = title
        self.bar_width = bar_width
        
        self.x = np.arange(len(FOOTSIES_ACTION_MOVES))

        # DPG items
        self.x_axis = None
        self.y_axis = None
        self.predicted_series = None
        self.estimated_series = None
    
    def setup(self):
        y = np.zeros((len(FOOTSIES_ACTION_MOVES),))

        with dpg.plot(label=self.title, width=1200):
            dpg.add_plot_legend()
            
            self.x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Move")
            dpg.set_axis_ticks(self.x_axis, tuple([(move.name, i + 1) for i, move in enumerate(FOOTSIES_ACTION_MOVES)]))
            dpg.set_axis_limits(self.x_axis, 0.0, len(FOOTSIES_ACTION_MOVES) + 2) # the + 2 is padding

            self.y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Probability")
            dpg.set_axis_limits(self.y_axis, 0.0, 1.0)
        
            self.predicted_series = dpg.add_bar_series(self.x + 1.0 - self.bar_width, y, label="Predicted", weight=self.bar_width * 2, parent=self.y_axis)
            self.estimated_series = dpg.add_bar_series(self.x + 1.0 + self.bar_width, y, label="Estimated", weight=self.bar_width * 2, parent=self.y_axis)

    def update(self, distribution_predicted, distribution_estimated):
        dpg.set_value(self.predicted_series, [list(self.x + 1.0 - self.bar_width), list(distribution_predicted)])
        dpg.set_value(self.estimated_series, [list(self.x + 1.0 + self.bar_width), list(distribution_estimated)])


class MimicAnalyserManager:
    def __init__(self):
        self.p1_plot = OpponentDistributionPlot("Player 1 move probability distribution")
        self.p2_plot = OpponentDistributionPlot("Player 2 move probability distribution")

        n_moves = len(FOOTSIES_ACTION_MOVES)
        self.p1_action_table = ActionTableEstimator(n_moves)
        self.p2_action_table = ActionTableEstimator(n_moves)

        # DPG items
        self.actual_p1_move = None
        self.actual_p2_move = None
        self.p1_action_table_estimator_size = None
        self.p2_action_table_estimator_size = None

    def include_mimic_dpg_elements(self, analyser: Analyser):
        dpg.add_text("Opponent model estimations on the previous observation")
        
        self.p1_plot.setup()
        self.p2_plot.setup()

        dpg.add_separator()

        with dpg.group(horizontal=True):
            dpg.add_text("Size of action table estimator of player 1: ")
            self.p1_action_table_estimator_size = dpg.add_text("")
        
        with dpg.group(horizontal=True):
            dpg.add_text("Size of action table estimator of player 2: ")
            self.p2_action_table_estimator_size = dpg.add_text("")

        with dpg.group(horizontal=True):
            dpg.add_text("Actual move of player 1:")
            self.actual_p1_move = dpg.add_text("")
        
        with dpg.group(horizontal=True):
            dpg.add_text("Actual move of player 2:")
            self.actual_p2_move = dpg.add_text("")

    def predict_next_move(self, analyser: Analyser):
        if analyser.previous_observation is None:
            return

        agent: OpponentModelAgent = analyser.agent
        observation = analyser.previous_observation
        p1_move = FOOTSIES_ACTION_MOVE_INDICES_MAP[analyser.current_info["p1_move"]]
        p2_move = FOOTSIES_ACTION_MOVE_INDICES_MAP[analyser.current_info["p2_move"]]

        dpg.set_value(self.actual_p1_move, FOOTSIES_ACTION_MOVES[p1_move].name)
        dpg.set_value(self.actual_p2_move, FOOTSIES_ACTION_MOVES[p2_move].name)

        self.p1_action_table.update(observation, p1_move)
        self.p2_action_table.update(observation, p2_move)

        observation_torch = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        p1_distribution_predicted = agent.p1_model.probability_distribution(observation_torch).squeeze()
        p2_distribution_predicted = agent.p2_model.probability_distribution(observation_torch).squeeze()
        p1_distribution_estimated = self.p1_action_table.probability_distribution(observation)
        p2_distribution_estimated = self.p2_action_table.probability_distribution(observation)

        self.p1_plot.update(p1_distribution_predicted, p1_distribution_estimated)
        self.p2_plot.update(p2_distribution_predicted, p2_distribution_estimated)

        dpg.set_value(self.p1_action_table_estimator_size, str(self.p1_action_table.size()))
        dpg.set_value(self.p2_action_table_estimator_size, str(self.p2_action_table.size()))


if __name__ == "__main__":

    footsies_env = FootsiesEnv(
        game_path="../Footsies-Gym/Build/FOOTSIES.x86_64",
        game_port=15000,
        opponent_port=15001,
        render_mode="human",
        sync_mode="synced_non_blocking",
        fast_forward=False,
    )

    env = FootsiesActionCombinationsDiscretized(
        FlattenObservation(
            FootsiesNormalized(footsies_env)
        )
    )

    agent = OpponentModelAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        by_primitive_actions=False,
        use_sigmoid_output=False,
        input_clip=False,
        hidden_layer_sizes_specification="",
        hidden_layer_activation_specification="Identity",
    )

    load_agent_model(agent, "mimic_linear")

    mimic_analyser_manager = MimicAnalyserManager()

    analyser = Analyser(
        env=env,
        agent=agent,
        custom_elements_callback=mimic_analyser_manager.include_mimic_dpg_elements,
        custom_state_update_callback=mimic_analyser_manager.predict_next_move,
    )
    analyser.start()
