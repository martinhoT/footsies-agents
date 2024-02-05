import dearpygui.dearpygui as dpg
import numpy as np
import torch
from gymnasium.wrappers.flatten_observation import FlattenObservation
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.normalization import FootsiesNormalized
from analysis import Analyser
from agents.mimic.agent import FootsiesAgent as OpponentModelAgent
from agents.action import ActionMap
from main import load_agent_model


AGENT: OpponentModelAgent = None


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
        
        self.x = np.arange(ActionMap.n_simple())

        # DPG items
        self.x_axis = None
        self.y_axis = None
        self.predicted_series = None
        self.estimated_series = None
    
    def setup(self):
        y = np.zeros((ActionMap.n_simple(),))

        with dpg.plot(label=self.title, width=1200):
            dpg.add_plot_legend()
            
            self.x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Move")
            dpg.set_axis_ticks(self.x_axis, tuple([(move.name, i + 1) for i, move in enumerate(ActionMap.SIMPLE_ACTIONS)]))
            dpg.set_axis_limits(self.x_axis, 0.0, ActionMap.n_simple() + 2) # the + 2 is padding

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

        n_moves = ActionMap.n_simple()
        self.p1_action_table = ActionTableEstimator(n_moves)
        self.p2_action_table = ActionTableEstimator(n_moves)
        
        # In case we want to act with the predictions, since they are done as simple actions
        self.p2_predicted_action_iterator = iter([])

        # DPG items
        self.actual_p1_move = None
        self.actual_p2_move = None
        self.p1_action_table_estimator_size = None
        self.p2_action_table_estimator_size = None
        self.p1_frameskip = None
        self.p2_frameskip = None

    @property
    def online_learning(self) -> bool:
        return AGENT.learn_p1 or AGENT.learn_p2

    def toggle_p1_online_learning(self):
        AGENT.learn_p1 = not AGENT.learn_p1
    
    def toggle_p2_online_learning(self):
        AGENT.learn_p2 = not AGENT.learn_p2

    def update_max_loss(self, max_loss: float):
        AGENT.p1_model.max_loss = max_loss
        AGENT.p2_model.max_loss = max_loss

    def include_mimic_dpg_elements(self, analyser: Analyser):
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="Enable P1 online learning", default_value=AGENT.learn_p1, callback=self.toggle_p1_online_learning)
            dpg.add_checkbox(label="Enable P2 online learning", default_value=AGENT.learn_p2, callback=self.toggle_p2_online_learning)
            dpg.add_input_float(label="Max. allowed loss", default_value=AGENT.p1_model.max_loss, width=100, callback=lambda s, a: self.update_max_loss(a))
            self.p1_frameskip = dpg.add_checkbox(label="P1 frameskip", default_value=False, enabled=False)
            self.p2_frameskip = dpg.add_checkbox(label="P2 frameskip", default_value=False, enabled=False)

        dpg.add_separator()

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

        observation = analyser.previous_observation
        p1_move_state = ActionMap.move_from_move_index(analyser.current_info["p1_move"])
        p2_move_state = ActionMap.move_from_move_index(analyser.current_info["p2_move"])
        p1_simple = ActionMap.simple_from_move(p1_move_state)
        p2_simple = ActionMap.simple_from_move(p2_move_state)
        p1_simple_move = ActionMap.simple_as_move(p1_simple)
        p2_simple_move = ActionMap.simple_as_move(p2_simple)

        dpg.set_value(self.actual_p1_move, p1_simple_move.name)
        dpg.set_value(self.actual_p2_move, p2_simple_move.name)

        dpg.set_value(self.p1_frameskip, not ActionMap.is_state_actionable_late(p1_move_state, analyser.previous_observation[32], analyser.current_observation[32]))
        dpg.set_value(self.p2_frameskip, not ActionMap.is_state_actionable_late(p2_move_state, analyser.previous_observation[33], analyser.current_observation[33]))

        self.p1_action_table.update(observation, p1_simple)
        self.p2_action_table.update(observation, p2_simple)

        if self.online_learning:
            # We need to call act so that the agent can store the current observation. Implementation detail, but whatever
            AGENT.act(analyser.previous_observation, analyser.previous_info)
            AGENT.update(analyser.current_observation, None, None, None, analyser.current_info)

        observation_torch = AGENT._obs_to_tensor(observation)
        p1_distribution_predicted = AGENT.p1_model.probability_distribution(AGENT.p1_model.mask_environment_observation(observation_torch)).squeeze()
        p2_distribution_predicted = AGENT.p2_model.probability_distribution(AGENT.p2_model.mask_environment_observation(observation_torch)).squeeze()
        p1_distribution_estimated = self.p1_action_table.probability_distribution(observation)
        p2_distribution_estimated = self.p2_action_table.probability_distribution(observation)

        self.p1_plot.update(p1_distribution_predicted, p1_distribution_estimated)
        self.p2_plot.update(p2_distribution_predicted, p2_distribution_estimated)

        dpg.set_value(self.p1_action_table_estimator_size, str(self.p1_action_table.size()))
        dpg.set_value(self.p2_action_table_estimator_size, str(self.p2_action_table.size()))

    def p2_prediction_discrete(self, obs: np.ndarray, info: dict) -> int:
        try:
            discrete_action = next(self.p2_predicted_action_iterator)
        except StopIteration:
            obs = AGENT._obs_to_tensor(obs)
            
            # We need to craft the observation to be as if P1 is the one experiencing it
            #  guard
            obs[:, [0, 1]] = obs[:, [1, 0]]
            #  move
            tmp = obs[:, 2:17].clone().detach()
            obs[:, 2:17] = obs[:, 17:32]
            obs[:, 17:32] = tmp
            #  move progress
            obs[:, [32, 33]] = obs[:, [33, 32]]
            #  position
            obs[:, [34, 35]] = obs[:, [35, 34]]
            
            simple_action = AGENT.p2_model.predict(AGENT.p2_model.mask_environment_observation(obs), deterministic=True).item()
            print(simple_action)
            self.p2_predicted_action_iterator = iter(ActionMap.simple_to_discrete(simple_action))
            discrete_action = next(self.p2_predicted_action_iterator)
        
        return discrete_action


if __name__ == "__main__":

    footsies_env = FootsiesEnv(
        game_path="../Footsies-Gym/Build/FOOTSIES.x86_64",
        game_port=15000,
        opponent_port=15001,
        remote_control_port=15002,
        render_mode="human",
        sync_mode="synced_non_blocking",
        fast_forward=False,
        vs_player=True,
    )

    env = FootsiesActionCombinationsDiscretized(
        FlattenObservation(
            FootsiesNormalized(footsies_env)
        )
    )

    AGENT = OpponentModelAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        frameskipping=True,
        by_primitive_actions=False,
        use_sigmoid_output=False,
        input_clip=False,
        input_clip_leaky_coef=0.01,
        max_allowed_loss=float("+inf"),
        hidden_layer_sizes_specification="",
        hidden_layer_activation_specification="Identity",
        mini_batch_size=1,
        learning_rate=0.005,
        move_transition_scale=30,
    )

    # load_agent_model(AGENT, "mimic_linear_frameskip")

    mimic_analyser_manager = MimicAnalyserManager()

    analyser = Analyser(
        env=env,
        p1_action_source=mimic_analyser_manager.p2_prediction_discrete,
        custom_elements_callback=mimic_analyser_manager.include_mimic_dpg_elements,
        custom_state_update_callback=mimic_analyser_manager.predict_next_move,
    )
    analyser.start()
