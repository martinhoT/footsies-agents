from typing import Iterable
import torch
import dearpygui.dearpygui as dpg
import numpy as np
from torch import nn
from gymnasium.wrappers.flatten_observation import FlattenObservation
from gymnasium.wrappers.transform_observation import TransformObservation
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.moves import FootsiesMove
from analysis import Analyser
from agents.mimic.agent import FootsiesAgent as OpponentModelAgent, PlayerModel, PlayerModelNetwork
from agents.action import ActionMap
from main import load_agent_model
from agents.torch_utils import observation_invert_perspective_flattened


# It's just a glorified dictionary
class ActionTableEstimator:
    def __init__(
        self,
        n_moves: int,
    ):
        self.n_moves = n_moves
        self.table = {}
    
    def update(self, observation: torch.Tensor, move):
        obs = observation.numpy(force=True).tobytes()
        if obs not in self.table:
            self.table[obs] = np.zeros((self.n_moves,))
        
        self.table[obs][move] += 1
    
    def probability_distribution(self, observation: torch.Tensor) -> np.ndarray:
        obs = observation.numpy().tobytes()
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

    def update(self, distribution_predicted: np.ndarray, distribution_estimated: np.ndarray):
        if distribution_predicted is not None:
            dpg.set_value(self.predicted_series, [list(self.x + 1.0 - self.bar_width), list(distribution_predicted)])
        if distribution_estimated is not None:
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


class LossPlot:
    def __init__(
        self,
        title: str,
        max_view_steps: int = 50,
        threshold: int = 0.0,
    ):
        self.title = title
        self.max_view_steps = max_view_steps
        self.threshold = threshold

        self.x = list(range(max_view_steps))
        self.y_current_loss = [0.0] * max_view_steps
        self.y_smoothed_loss = [0.0] * max_view_steps
        self.y_threshold = [0.0] * max_view_steps
        self.current_step = 0

        self.y_max = float("-inf")
        self.y_min = float("+inf")

        # DPG items
        self.x_axis = None
        self.y_axis = None
    
    def setup(self, width: int = 1200):
        with dpg.plot(label=self.title, width=width):
            dpg.add_plot_legend()

            self.x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Episode")
            dpg.set_axis_limits(self.x_axis, 0, self.max_view_steps - 1)

            self.y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Loss")
            dpg.set_axis_limits_auto(self.y_axis)

            self.current_loss_plot = dpg.add_line_series(self.x, self.y_current_loss, label="Loss at this example", parent=self.y_axis)
            self.smoothed_loss_plot = dpg.add_line_series(self.x, self.y_smoothed_loss, label="Smoothed loss", parent=self.y_axis)
            self.threshold_plot = dpg.add_line_series(self.x, self.y_threshold, label="Threshold", parent=self.y_axis)
    
    def update(self, new_loss: float, smoothed_loss: float, new_threshold: float = None):
        if new_threshold is not None:
            self.threshold = new_threshold
        y_threshold = (1 + self.threshold) * smoothed_loss

        self.y_max = max(self.y_max, new_loss, y_threshold, smoothed_loss)
        self.y_min = min(self.y_min, new_loss, y_threshold, smoothed_loss)
        self.y_current_loss[self.current_step] = new_loss
        self.y_smoothed_loss[self.current_step] = smoothed_loss
        self.y_threshold[self.current_step] = y_threshold
        self.current_step = (self.current_step + 1) % self.max_view_steps

        dpg.set_value(self.current_loss_plot, [list(self.x), list(self.y_current_loss)])
        dpg.set_value(self.smoothed_loss_plot, [list(self.x), list(self.y_smoothed_loss)])
        dpg.set_value(self.threshold_plot, [list(self.x), list(self.y_threshold)])
        dpg.set_axis_limits(self.y_axis, self.y_min, self.y_max)


class MimicAnalyserManager:
    def __init__(
        self,
        p1_model: PlayerModel | None,
        p2_model: PlayerModel | None,
        p1_mirror_p2: bool = False
    ):
        if p1_model is None and p2_model is None:
            raise ValueError("at least one model should be used, or else the analyser is useless")
        
        self.p1_model = p1_model
        self.p2_model = p2_model
        self._p1_mirror_p2 = p1_mirror_p2

        self._learn_p1 = p1_model is not None
        self._learn_p2 = p2_model is not None

        self.p1_plot = OpponentDistributionPlot("Player 1 move probability distribution")
        self.p2_plot = OpponentDistributionPlot("Player 2 move probability distribution")
        
        # NOTE: Recommended to use odd bins so that one bin catches values around 0
        self.p1_gradient_plot = GradientHistogramPlot("Player 1 learning gradients", 101, -1, 1)
        self.p2_gradient_plot = GradientHistogramPlot("Player 2 learning gradients", 101, -1, 1)

        self.p1_loss_plot = LossPlot("Player 1 learning loss", 50, self.p1_model.scar_detection_threshold)
        self.p2_loss_plot = LossPlot("Player 2 learning loss", 50, self.p2_model.scar_detection_threshold)

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
        self.p1_scar_size = None
        self.p2_scar_size = None
        self.attribute_modifier_window = None

    @property
    def learn_p1(self) -> bool:
        return self._learn_p1
    
    @property
    def learn_p2(self) -> bool:
        return self._learn_p2

    @property
    def p1_mirror_p2(self) -> bool:
        return self._p1_mirror_p2

    def toggle_learn_p1(self):
        if self.p1_model is None:
            self._learn_p1 = False
            return
        
        self._learn_p1 = not self._learn_p1

    def toggle_learn_p2(self):
        if self.p2_model is None:
            self._learn_p2 = False
            return

        self._learn_p2 = not self._learn_p2

    def toggle_p1_mirror_p2(self):
        self._p1_mirror_p2 = not self._p1_mirror_p2

    def include_mimic_dpg_elements(self, analyser: Analyser):
        with dpg.group(horizontal=True):
            if self.p1_model is not None:
                dpg.add_checkbox(label="Enable P1 online learning", default_value=self.learn_p1, callback=self.toggle_learn_p1)
            if self.p2_model is not None:
                dpg.add_checkbox(label="Enable P2 online learning", default_value=self.learn_p2, callback=self.toggle_learn_p2)
            dpg.add_checkbox(label="P1 mirror P2", default_value=self._p1_mirror_p2, callback=self.toggle_p1_mirror_p2)
            self.p1_frameskip = dpg.add_checkbox(label="P1 frameskip", default_value=False, enabled=False)
            self.p2_frameskip = dpg.add_checkbox(label="P2 frameskip", default_value=False, enabled=False)

        dpg.add_separator()

        dpg.add_text("Opponent model estimations on the previous observation")
        
        self.p1_plot.setup(width=1050)
        self.p2_plot.setup(width=1050)

        dpg.add_separator()

        if self.learn_p1:
            with dpg.group(horizontal=True):
                dpg.add_text("Size of action table estimator of player 1: ")
                self.p1_action_table_estimator_size = dpg.add_text("")
        
        if self.learn_p2:
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

        with dpg.window(label="Attribute modifier", show=False) as self.attribute_modifier_window:
            if self.learn_p1:
                dpg.add_text("Player 1")
                dpg.add_slider_float(label="Scar detection threshold", default_value=self.p1_model.scar_detection_threshold, max_value=1.0, min_value=0.0, width=200, enabled=True, callback=lambda s, a: setattr(self.p1_model, "scar_detection_threshold", a))
                dpg.add_slider_float(label="Smoothed loss coef", default_value=self.p1_model.smoothed_loss_coef, max_value=1.0, min_value=0.0, width=200, enabled=True, callback=lambda s, a: setattr(self.p1_model, "smoothed_loss_coef", a))
                dpg.add_slider_float(label="Move transition scale", default_value=self.p1_model.move_transition_scale, max_value=1000.0, min_value=1.0, width=200, enabled=True, callback=lambda s, a: setattr(self.p1_model, "move_transition_scale", a))
                dpg.add_slider_float(label="Learning rate", default_value=self.p1_model.optimizer.param_groups[0]["lr"], max_value=1e-2, min_value=1e-6, width=200, enabled=True, callback=lambda s, a: self.p1_model.set_learning_rate(a))

            else:
                dpg.add_text("No player 1 model")

            dpg.add_separator()

            if self.learn_p2:
                dpg.add_text("Player 2")
                dpg.add_slider_float(label="Scar detection threshold", default_value=self.p2_model.scar_detection_threshold, max_value=1.0, min_value=0.0, width=200, enabled=True, callback=lambda s, a: setattr(self.p2_model, "scar_detection_threshold", a))
                dpg.add_slider_float(label="Smoothed loss coef", default_value=self.p2_model.smoothed_loss_coef, max_value=1.0, min_value=0.0, width=200, enabled=True, callback=lambda s, a: setattr(self.p2_model, "smoothed_loss_coef", a))
                dpg.add_slider_float(label="Move transition scale", default_value=self.p2_model.move_transition_scale, max_value=1000.0, min_value=1.0, width=200, enabled=True, callback=lambda s, a: setattr(self.p2_model, "move_transition_scale", a))
                dpg.add_slider_float(label="Learning rate", default_value=self.p2_model.optimizer.param_groups[0]["lr"], max_value=1e-2, min_value=1e-6, width=200, enabled=True, callback=lambda s, a: self.p2_model.set_learning_rate(a))
            
            else:
                dpg.add_text("No player 2 model")

        dpg.add_button(label="Open attribute modifier", callback=lambda: dpg.show_item(self.attribute_modifier_window))

        with dpg.collapsing_header(label="Loss plots"):
            with dpg.group(horizontal=True):
                with dpg.group():
                    self.p1_loss_plot.setup(525)
                    self.p1_scar_size = dpg.add_slider_float(label="Player 1 scar size", default_value=len(self.p1_model.x_batch_as_list), max_value=100, min_value=0, width=100, enabled=False)
                with dpg.group():
                    self.p2_loss_plot.setup(525)
                    self.p2_scar_size = dpg.add_slider_float(label="Player 2 scar size", default_value=len(self.p2_model.x_batch_as_list), max_value=100, min_value=0, width=100, enabled=False)
            
        with dpg.collapsing_header(label="Gradient plots"):
            with dpg.group(horizontal=True):
                self.p1_gradient_plot.setup(525)
                self.p2_gradient_plot.setup(525)
        
    def predict_next_move(self, analyser: Analyser):
        if analyser.previous_observation is None:
            return

        obs = analyser.previous_observation
        next_obs = analyser.current_observation
        p1_simple, p2_simple = ActionMap.simples_from_torch_transition(obs, next_obs)
        
        # Determine which actions the players did
        if p1_simple is not None:
            p1_simple_move = ActionMap.simple_as_move(p1_simple)
            dpg.set_value(self.actual_p1_move, p1_simple_move.name)
        
        if p2_simple is not None:
            p2_simple_move = ActionMap.simple_as_move(p2_simple)
            dpg.set_value(self.actual_p2_move, p2_simple_move.name)

        # Update whether the actions were ineffectual (frameskipped)
        dpg.set_value(self.p1_frameskip, p1_simple is None)
        dpg.set_value(self.p2_frameskip, p2_simple is None)

        # Update action tables
        self.p1_action_table.update(obs, p1_simple)
        self.p2_action_table.update(obs, p2_simple)

        # Update P1 model
        if self.learn_p1:
            self.p1_model.update(obs, p1_simple)

            self.p1_gradient_plot.update(self.p1_model.network.parameters())
            self.p1_loss_plot.update(self.p1_model.most_recent_loss, self.p1_model.smoothed_loss, self.p1_model.scar_detection_threshold)
            dpg.set_value(self.p1_scar_size, len(self.p1_model.x_batch_as_list))
        
        # Update P2 model
        if self.learn_p2:
            self.p2_model.update(obs, p2_simple)
            
            self.p2_gradient_plot.update(self.p2_model.network.parameters())
            self.p2_loss_plot.update(self.p2_model.most_recent_loss, self.p2_model.smoothed_loss, self.p2_model.scar_detection_threshold)
            dpg.set_value(self.p2_scar_size, len(self.p2_model.x_batch_as_list))
        
        # Get probability distribution of P1 model
        p1_distribution_predicted = None
        if self.p1_mirror_p2:
            obs_inverted = observation_invert_perspective_flattened(obs)
            p1_distribution_predicted = self.p2_model.probability_distribution(obs_inverted).squeeze()
        elif self.p1_model is not None:
            p1_distribution_predicted = self.p1_model.probability_distribution(obs).squeeze()
        
        # Get probability distribution of P2 model
        p2_distribution_predicted = None
        if self.p2_model is not None:
            p2_distribution_predicted = self.p2_model.probability_distribution(obs).squeeze()

        # Get probability distributions from the action tables
        p1_distribution_estimated = self.p1_action_table.probability_distribution(obs) # we don't mirror this one, it's not that important
        p2_distribution_estimated = self.p2_action_table.probability_distribution(obs)

        # Update the plots and other display variables        
        self.p1_plot.update(p1_distribution_predicted, p1_distribution_estimated)
        self.p2_plot.update(p2_distribution_predicted, p2_distribution_estimated)
        dpg.set_value(self.p1_action_table_estimator_size, str(self.p1_action_table.size()))
        dpg.set_value(self.p2_action_table_estimator_size, str(self.p2_action_table.size()))

    def p2_prediction_discrete(self, obs: torch.Tensor, info: dict) -> int:
        if not self.p1_mirror_p2:
            return 0
        
        try:
            discrete_action = next(self.p2_predicted_action_iterator)
        except StopIteration:
            # We need to craft the observation to be as if P1 is the one experiencing it (since P2 model was trained inverted)
            obs_inverted = observation_invert_perspective_flattened(obs)
            probs = self.p2_model.probability_distribution(obs_inverted).squeeze()
            distribution = torch.distributions.Categorical(probs=probs)
            action = distribution.sample().item()
            self.p2_predicted_action_iterator = iter(ActionMap.simple_to_discrete(action))
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

    env = TransformObservation(
        FootsiesActionCombinationsDiscretized(
            FlattenObservation(
                FootsiesNormalized(footsies_env)
            )
        ),
        lambda o: torch.from_numpy(o).float().unsqueeze(0)
    )

    obs_dim = env.observation_space.shape[0]
    action_dim = ActionMap.n_simple()

    player_model = lambda: PlayerModel(
        obs_size=obs_dim,
        player_model_network=PlayerModelNetwork(
            input_dim=obs_dim,
            output_dim=action_dim,
            use_sigmoid_output=False,
            input_clip=False,
            input_clip_leaky_coef=0,
            hidden_layer_sizes=[64, 64],
            hidden_layer_activation=nn.LeakyReLU,
        ),
        mini_batch_size=1,
        move_transition_scale=1.0,
        learning_rate=3e-3,
        reinforce_max_loss=float("+inf"),
        reinforce_max_iters=float("+inf"),
        scar_max_size=1000,
        scar_loss_coef=1.0,
        scar_recency_coef=0.0,
        scar_detection_threshold=0.1,
        smoothed_loss_coef=0.8,
    )

    # Create copies of the same model
    p1_model = player_model()
    p2_model = player_model()

    agent = OpponentModelAgent(
        observation_space_size=obs_dim,
        action_space_size=action_dim,
        tile_coding=False,
        append_last_actions_n=0,
        append_last_actions_distinct=False,
        by_primitive_actions=False,
        p1_model=p1_model,
        p2_model=p2_model,
    )

    # load_agent_model(agent, "mimic_linear_frameskip")

    mimic_analyser_manager = MimicAnalyserManager(
        p1_model=None,
        p2_model=agent,
        p1_mirror_p2=False,
    )

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
