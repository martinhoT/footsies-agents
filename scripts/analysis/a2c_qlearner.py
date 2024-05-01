import torch
import dearpygui.dearpygui as dpg
import numpy as np
from typing import cast
from gymnasium.wrappers.flatten_observation import FlattenObservation
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.moves import FootsiesMove
from scripts.analysis.base import Analyser
from agents.action import ActionMap
from agents.a2c.a2c import A2CQLearner
from agents.a2c.agent import A2CAgent
from agents.ql.ql import QFunctionNetwork, QFunctionTable
from main import import_agent, load_agent, load_agent_parameters


SIMPLE_ACTION_LABELS_GEN = lambda labels, n: tuple((move.name, i / n + (1 / (n * 2))) for i, move in enumerate(labels))
SIMPLE_ACTION_LABELS_REVERSED_GEN = lambda labels, n: tuple((move.name, (n - 1) / n - i / n + (1 / (n * 2))) for i, move in enumerate(labels))

SIMPLE_ACTION_LABELS = SIMPLE_ACTION_LABELS_GEN(ActionMap.SIMPLE_ACTIONS, ActionMap.n_simple())
SIMPLE_ACTION_LABELS_REVERSED = SIMPLE_ACTION_LABELS_REVERSED_GEN(ActionMap.SIMPLE_ACTIONS, ActionMap.n_simple())


class PlayerActionMatrix:
    def __init__(
        self,
        action_dim: int,
        opponent_action_dim: int,
        add_color_scale: bool = True,
        auto_scale: bool = False,
    ):
        self.action_dim = action_dim
        self.opponent_action_dim = opponent_action_dim
        self.add_color_scale = add_color_scale
        self.auto_scale = auto_scale
        self.color_scale: int | str = ""
        self.series: int | str = ""

    def setup(self, title: str, value_range: tuple[float, float] = (-1.0, 1.0)):
        sx, sy = value_range
        agent_actions = SIMPLE_ACTION_LABELS_GEN(ActionMap.SIMPLE_ACTIONS[:self.action_dim], self.action_dim)
        opponent_actions = SIMPLE_ACTION_LABELS_REVERSED_GEN(ActionMap.SIMPLE_ACTIONS[:self.opponent_action_dim], self.opponent_action_dim)

        with dpg.group(horizontal=True):
            if self.add_color_scale:
                self.color_scale = dpg.add_colormap_scale(min_scale=sx, max_scale=sy, colormap=dpg.mvPlotColormap_Viridis, height=400)

            with dpg.plot(label=title, no_mouse_pos=True, height=400, width=-1) as plot:
                dpg.bind_colormap(plot, dpg.mvPlotColormap_Viridis)
                x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Agent action", lock_min=True, lock_max=True, no_gridlines=True, no_tick_marks=True)
                dpg.set_axis_ticks(x_axis, agent_actions)

                with dpg.plot_axis(dpg.mvYAxis, label="Opponent action", lock_min=True, lock_max=True, no_gridlines=True, no_tick_marks=True) as y_axis:
                    dpg.set_axis_ticks(y_axis, opponent_actions)
                    initial_data = np.zeros((self.opponent_action_dim, self.action_dim))
                    self.series = dpg.add_heat_series(initial_data.flatten().tolist(), rows=self.opponent_action_dim, cols=self.action_dim, scale_min=sx, scale_max=sy, format="%0.2f")

    def update(self, values: np.ndarray):
        dpg.set_value(self.series, [values.flatten().tolist(), [self.opponent_action_dim, self.action_dim]])

        if self.auto_scale:
            mn, mx = np.min(values), np.max(values)
            if self.color_scale is not None:
                dpg.configure_item(self.color_scale, min_scale=mn, max_scale=mx)
            dpg.configure_item(self.series, scale_min=mn, scale_max=mx)


class PolicyDistributionPlot:
    def __init__(
        self,
        action_dim: int,
        bar_width: float = 0.2,
    ):
        self.bar_width = bar_width
        
        self.action_dim = action_dim
        self.x = np.arange(action_dim)

        # DPG items
        self.x_axis: int | str = ""
        self.y_axis: int | str = ""
        self.series: int | str = ""
    
    def setup(self, title: str = "Policy distribution", width: int = 1050):
        y = np.zeros((self.action_dim,))

        with dpg.plot(label=title, width=width) as plot:
            self.x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Move")
            dpg.set_axis_ticks(self.x_axis, tuple([(move.name, i + 1) for i, move in enumerate(ActionMap.SIMPLE_ACTIONS[:self.action_dim])]))
            dpg.set_axis_limits(self.x_axis, 0.0, self.action_dim + 1) # the + 1 is padding

            self.y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Probability")
            dpg.set_axis_limits(self.y_axis, 0.0, 1.0)
        
            self.series = dpg.add_bar_series((self.x + 1.0).tolist(), y.tolist(), weight=self.bar_width * 2, parent=self.y_axis)
            dpg.bind_colormap(plot, dpg.mvPlotColormap_Default)

    def update(self, distribution: np.ndarray):
        dpg.set_value(self.series, [list(self.x + 1.0), distribution.tolist()])


class QLearnerAnalyserManager:
    def __init__(
        self,
        agent: A2CAgent,
        action_dim: int,
        opponent_action_dim: int,
        include_online_learning: bool = True,
    ):
        self.agent = agent
        # Assume it's the QLearner to avoid headache
        self.learner: A2CQLearner = agent.learner
        self.action_dim = action_dim
        self.opponent_action_dim = opponent_action_dim
        self._include_online_learning = include_online_learning

        self.online_learning = False
        self.show_target_network = False

        self.q_table_plot = PlayerActionMatrix(action_dim, opponent_action_dim, auto_scale=False)
        self.q_table_update_frequency_plot = PlayerActionMatrix(action_dim, opponent_action_dim, add_color_scale=False, auto_scale=True) if isinstance(self.agent._critic, QFunctionTable) else None
        self.policy_plot = PlayerActionMatrix(action_dim, opponent_action_dim, auto_scale=False)
        
        self.current_observation = None

        # DPG items
        self.attribute_modifier_window: int | str = ""
        self.current_observation_index: int | str = ""
        self.current_action: int | str = ""

    def add_custom_elements(self, analyser: Analyser):
        with dpg.window(label="Attribute modifier", show=False) as self.attribute_modifier_window:
            dpg.add_slider_float(label="MaxEnt", default_value=self.learner.maxent, min_value=0.0, max_value=1.0, callback=lambda s, a: setattr(self.learner, "maxent", a))
            dpg.add_checkbox(label="MaxEnt gradient flow", default_value=self.learner.maxent_gradient_flow, callback=lambda s, a: setattr(self.learner, "maxent_gradient_flow", a))
            
            dpg.add_separator()
            
            dpg.add_text("Actor")
            dpg.add_slider_float(label="Learning rate", default_value=self.learner.actor_learning_rate, min_value=0.0, max_value=1.0, callback=lambda s, a: setattr(self.learner, "actor_learning_rate", a))
            dpg.add_slider_float(label="Entropy loss coef", default_value=self.learner.actor_entropy_loss_coef, min_value=0.0, max_value=1.0, callback=lambda s, a: setattr(self.learner, "actor_entropy_loss_coef", a))

            dpg.add_separator()

            dpg.add_text("Critic")
            dpg.add_slider_float(label="Learning rate", default_value=self.learner.critic_learning_rate, min_value=0.0, max_value=1.0, callback=lambda s, a: setattr(self.learner, "critic_learning_rate", a))
            dpg.add_slider_float(label="Discount", default_value=self.learner.critic.discount, min_value=0.0, max_value=1.0, callback=lambda s, a: setattr(self.learner.critic, "discount", a))

        with dpg.group(horizontal=True):
            dpg.add_button(label="Open attribute modifier", callback=lambda: dpg.show_item(self.attribute_modifier_window))
            if self._include_online_learning:
                dpg.add_checkbox(label="Online learning", default_value=self.online_learning, callback=lambda s, a: setattr(self, "online_learning", a))

        dpg.add_separator()

        if isinstance(self.learner.critic, QFunctionNetwork):
            dpg.add_checkbox(label="Show target network", default_value=self.show_target_network, callback=lambda s, a: setattr(self, "show_target_network", a))
        self.q_table_plot.setup(title="Q-table values", value_range=(-1.0, 1.0))
    
        if self.q_table_update_frequency_plot is not None:
            self.q_table_update_frequency_plot.setup(title="Q-table update frequency")        

        if isinstance(self.learner.critic, QFunctionTable):
            with dpg.group(horizontal=True):
                dpg.add_text("Current observation index: ")
                self.current_observation_index = dpg.add_text("0")

        dpg.add_separator()
        
        self.policy_plot.setup(title="Policy distributions", value_range=(0.0, 1.0))

        with dpg.group(horizontal=True):
            dpg.add_text("Current action:")
            self.current_action = dpg.add_text("")

    def update_policy_distribution(self, obs: torch.Tensor):
        policy_distribution = self.agent._actor.probabilities(obs, None).numpy(force=True).flatten()
        self.policy_plot.update(policy_distribution)

    def on_state_update(self, analyser: Analyser):
        obs = cast(torch.Tensor, analyser.current_observation)

        # Make the agent learn first before presenting results, it's less confusing and we can immediately see the results
        if self.online_learning and analyser.most_recent_transition is not None:
            obs, next_obs, reward, terminated, truncated, info, next_info = analyser.most_recent_transition.as_tuple()
            # Clone the observations, detached from the graph (clone() is differentiable, so we use detach()), 
            # don't get internal references in the analyser since those can change dynamically
            # (e.g. by loading battle states), which messes up the gradient computation graph
            obs = obs.detach().clone()
            next_obs = next_obs.detach().clone()

            if not isinstance(obs, torch.Tensor) or not isinstance(next_obs, torch.Tensor):
                raise RuntimeError("the online learning portion of the A2C analyser assumes the environment is providing PyTorch tensors as observations, but that is not the case!")

            self.agent.update(obs, next_obs, reward, terminated, truncated, info, next_info)

        if isinstance(self.learner.critic, QFunctionNetwork):
            q_values = self.learner.critic.q(obs, use_target_network=self.show_target_network)
        else:
            q_values = self.learner.critic.q(obs)
        self.q_table_plot.update(q_values.numpy(force=True))
        if self.q_table_update_frequency_plot is not None and isinstance(self.agent.learner.critic, QFunctionTable):
            update_frequency = self.agent.learner.critic.update_frequency(obs)
            self.q_table_update_frequency_plot.update(update_frequency.numpy(force=True))

        if isinstance(self.learner.critic, QFunctionTable) and self.current_observation_index is not None:
            dpg.set_value(self.current_observation_index, str(self.learner.critic._obs_idx(obs)))

        self.update_policy_distribution(obs)
        
        if self.agent.current_action is not None:
            dpg.set_value(self.current_action, ActionMap.simple_as_move(self.agent.current_action).name)


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

    parameters = load_agent_parameters("a2c_agent")
    agent = cast(A2CAgent, import_agent("a2c", env, parameters)[0])
    load_agent(agent, "a2c_qlearner_il")

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

    manager = QLearnerAnalyserManager(
        agent,
        action_dim=ActionMap.n_simple(),
        opponent_action_dim=ActionMap.n_simple(),
    )

    analyser = Analyser(
        env=env,
        # p1_action_source=lambda o, i: next(p1),
        p1_action_source=lambda o, i: agent.act(torch.from_numpy(o).float().unsqueeze(0), i, predicted_opponent_action=None),
        custom_elements_callback=manager.add_custom_elements,
        custom_state_update_callback=manager.on_state_update,
    )
    analyser.start(state_change_apply_immediately=True)
