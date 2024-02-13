from typing import Iterable
import torch
import dearpygui.dearpygui as dpg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as pltanim
from itertools import cycle
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from gymnasium.wrappers.flatten_observation import FlattenObservation
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.moves import FootsiesMove
from agents.torch_utils import observation_invert_perspective_flattened
from analysis import Analyser
from agents.mimic.agent import FootsiesAgent as OpponentModelAgent
from agents.action import ActionMap
from main import load_agent_model


AGENT: OpponentModelAgent = None
GRADIENT_HIST_BINS: int = 10


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
    
    def setup(self, width: int = 1200):
        y = np.zeros((ActionMap.n_simple(),))

        with dpg.plot(label=self.title, width=width):
            dpg.add_plot_legend()
            
            self.x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Move")
            dpg.set_axis_ticks(self.x_axis, tuple([(move.name, i + 1) for i, move in enumerate(ActionMap.SIMPLE_ACTIONS)]))
            dpg.set_axis_limits(self.x_axis, 0.0, ActionMap.n_simple() + 1) # the + 1 is padding

            self.y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Probability")
            dpg.set_axis_limits(self.y_axis, 0.0, 1.0)
        
            self.predicted_series = dpg.add_bar_series(self.x + 1.0 - self.bar_width, y, label="Predicted", weight=self.bar_width * 2, parent=self.y_axis)
            self.estimated_series = dpg.add_bar_series(self.x + 1.0 + self.bar_width, y, label="Estimated", weight=self.bar_width * 2, parent=self.y_axis)

    def update(self, distribution_predicted, distribution_estimated):
        dpg.set_value(self.predicted_series, [list(self.x + 1.0 - self.bar_width), list(distribution_predicted)])
        dpg.set_value(self.estimated_series, [list(self.x + 1.0 + self.bar_width), list(distribution_estimated)])


class GradientHistogramPlot:
    def __init__(
        self,
        title: str,
        bins: int = 10,
        start: float = -3.0,
        stop = 3.0,
    ):
        self.title = title
        
        step = (stop - start) / bins
        self.x = np.arange(start, stop, step)
        self.bins_edges = np.hstack((self.x, self.x[-1] + step))

        self.step = step
        self.start = start
        self.stop = stop

        # DPG items
        self.x_axis = None
        self.y_axis = None
        self.histogram = None
    
    def setup(self, width: int = 1200):
        with dpg.plot(label=self.title, width=width):
            self.x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Gradient")

            self.y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Density")
            dpg.set_axis_limits(self.y_axis, 0.0, 1.5)
        
            self.histogram = dpg.add_bar_series(self.x + self.step / 2, [0.0] * len(self.x), weight=self.step, parent=self.y_axis)

    def update(self, parameters: Iterable[torch.nn.Parameter]):
        grads = [param.grad.flatten() for param in parameters if param.grad is not None]
        if not grads:
            return
        
        hist, _ = np.histogram(torch.hstack(grads), bins=self.bins_edges, range=(self.start, self.stop), density=True)
        dpg.set_value(self.histogram, [self.x + self.step / 2, list(hist)])


class MimicAnalyserManager:
    def __init__(self, p1_mirror_p2: bool = False):
        self.p1_mirror_p2 = p1_mirror_p2

        self.p1_plot = OpponentDistributionPlot("Player 1 move probability distribution")
        self.p2_plot = OpponentDistributionPlot("Player 2 move probability distribution")
        
        # NOTE: Recommended to use odd bins so that one bin catches values around 0
        self.p1_gradient_plot = GradientHistogramPlot("Player 1 learning gradients", 101, -1, 1)
        self.p2_gradient_plot = GradientHistogramPlot("Player 2 learning gradients", 101, -1, 1)

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
        self.p1_loss_value = None
        self.p2_loss_value = None

    @property
    def online_learning(self) -> bool:
        return AGENT.learn_p1 or AGENT.learn_p2

    def toggle_p1_online_learning(self):
        AGENT.learn_p1 = not AGENT.learn_p1
    
    def toggle_p2_online_learning(self):
        AGENT.learn_p2 = not AGENT.learn_p2

    def toggle_plot_p1_gradient(self):
        self.plot_p1_gradient = not self.plot_p1_gradient

    def toggle_plot_p2_gradient(self):
        self.plot_p2_gradient = not self.plot_p2_gradient

    def toggle_p1_mirror_p2(self):
        self.p1_mirror_p2 = not self.p1_mirror_p2

    def include_mimic_dpg_elements(self, analyser: Analyser):
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="Enable P1 online learning", default_value=AGENT.learn_p1, callback=self.toggle_p1_online_learning)
            dpg.add_checkbox(label="Enable P2 online learning", default_value=AGENT.learn_p2, callback=self.toggle_p2_online_learning)
            dpg.add_checkbox(label="P1 mirror P2", default_value=self.p1_mirror_p2, callback=self.toggle_p1_mirror_p2)
            self.p1_frameskip = dpg.add_checkbox(label="P1 frameskip", default_value=False, enabled=False)
            self.p2_frameskip = dpg.add_checkbox(label="P2 frameskip", default_value=False, enabled=False)

        dpg.add_separator()

        dpg.add_text("Opponent model estimations on the previous observation")
        
        self.p1_plot.setup(width=1050)
        self.p2_plot.setup(width=1050)

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
        
        dpg.add_separator()

        dpg.add_text("Learning gradients")

        with dpg.group(horizontal=True):
            self.p1_gradient_plot.setup(500)
            self.p1_loss_value = dpg.add_slider_float(default_value=0.0, vertical=True, min_value=0.0, max_value=100.0, height=300)
            self.p2_gradient_plot.setup(500)
            self.p2_loss_value = dpg.add_slider_float(default_value=0.0, vertical=True, min_value=0.0, max_value=100.0, height=300)
        
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
            
            if AGENT.learn_p1:
                self.p1_gradient_plot.update(AGENT.p1_model.network.parameters())
                dpg.set_value(self.p1_loss_value, AGENT.evaluate_average_loss_and_clear(True))
            if AGENT.learn_p2:
                self.p2_gradient_plot.update(AGENT.p2_model.network.parameters())
                dpg.set_value(self.p2_loss_value, AGENT.evaluate_average_loss_and_clear(False))

        # Update the action histories even when not performing online learning
        else:
            AGENT.append_to_action_history(p1_simple, True)
            AGENT.append_to_action_history(p2_simple, False)

        observation_torch = AGENT._obs_to_tensor(observation)
        if self.p1_mirror_p2:
            inverted = observation_invert_perspective_flattened(observation_torch)
            p1_distribution_predicted = AGENT.p2_model.probability_distribution(AGENT.craft_observation(inverted, False, True)).squeeze()
        else:
            p1_distribution_predicted = AGENT.p1_model.probability_distribution(AGENT.craft_observation(observation_torch, True, True)).squeeze()
        p1_distribution_estimated = self.p1_action_table.probability_distribution(observation) # we don't mirror this one, it's not that important
        p2_distribution_predicted = AGENT.p2_model.probability_distribution(AGENT.craft_observation(observation_torch, False, False)).squeeze()
        p2_distribution_estimated = self.p2_action_table.probability_distribution(observation)

        self.p1_plot.update(p1_distribution_predicted, p1_distribution_estimated)
        self.p2_plot.update(p2_distribution_predicted, p2_distribution_estimated)

        dpg.set_value(self.p1_action_table_estimator_size, str(self.p1_action_table.size()))
        dpg.set_value(self.p2_action_table_estimator_size, str(self.p2_action_table.size()))

    def p2_prediction_discrete(self, obs: np.ndarray, info: dict) -> int:
        if not self.p1_mirror_p2:
            return 0
        
        try:
            discrete_action = next(self.p2_predicted_action_iterator)
        except StopIteration:
            obs = AGENT._obs_to_tensor(obs)
            
            # We need to craft the observation to be as if P1 is the one experiencing it (since P2 model was trained inverted)
            obs = observation_invert_perspective_flattened(obs)
            
            simple_action = AGENT.p2_model.predict(AGENT.craft_observation(obs, use_p1_model=False, use_p1_action_history=True), deterministic=True).item()
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
        observation_space_size=env.observation_space.shape[0],
        action_space_size=env.action_space.n,
        frameskipping=True,
        append_last_actions_n=0,
        append_last_actions_distinct=False,
        by_primitive_actions=False,
        use_sigmoid_output=False,
        input_clip=True,
        input_clip_leaky_coef=0.01,
        hidden_layer_sizes_specification="",
        hidden_layer_activation_specification="LeakyReLU",
        mini_batch_size=1,
        learning_rate=0.01,
        move_transition_scale=1,
        reinforce_max_loss=1.0,
        reinforce_max_iters=float("+inf"),
    )

    # load_agent_model(AGENT, "mimic_linear_frameskip")

    mimic_analyser_manager = MimicAnalyserManager()

    def spammer():
        from itertools import cycle
        for action in cycle((
            ActionMap.simple_to_discrete(ActionMap.SIMPLE_ACTIONS.index(FootsiesMove.N_ATTACK))[0],
            ActionMap.simple_to_discrete(ActionMap.SIMPLE_ACTIONS.index(FootsiesMove.STAND))[0]
        )):
            yield action

    def idle():
        while True:
            yield 0

    p1 = idle()

    analyser = Analyser(
        env=env,
        # p1_action_source=mimic_analyser_manager.p2_prediction_discrete,
        p1_action_source=lambda o, i: next(p1),
        custom_elements_callback=mimic_analyser_manager.include_mimic_dpg_elements,
        custom_state_update_callback=mimic_analyser_manager.predict_next_move,
    )
    analyser.start()
