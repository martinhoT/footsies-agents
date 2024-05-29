import torch
import dearpygui.dearpygui as dpg
from gymnasium.wrappers.flatten_observation import FlattenObservation
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.normalization import FootsiesNormalized
from scripts.analysis.base import Analyser
from agents.action import ActionMap
from main import load_agent
from scripts.analysis.a2c_qlearner import QLearnerAnalyserManager
from scripts.analysis.opponent_model import MimicAnalyserManager
from scripts.analysis.game_model import GameModelAnalyserManager
from models import to_
from gymnasium.wrappers.transform_observation import TransformObservation
from main import setup_logger, load_agent, load_agent_parameters, import_agent
from opponents.curriculum import CurriculumOpponent
from agents.wrappers import FootsiesSimpleActions
from agents.logger import TrainingLoggerWrapper
from agents.the_one.agent import TheOneAgent
from typing import Literal, cast
from gymnasium.spaces import Discrete
from scripts.analysis.base import editable_dpg_value
from os import path
from opponents import curriculum
from opponents.base import Opponent
import tyro
import logging
import warnings


class TheOneAnalyserManager:
    def __init__(self,
        agent: TheOneAgent | TrainingLoggerWrapper[TheOneAgent],
        qlearner_manager: QLearnerAnalyserManager,
        mimic_manager: MimicAnalyserManager | None,
        game_model_manager: GameModelAnalyserManager | None,
        custom_opponent: Opponent | None = None,
    ):
        self.agent = agent
        self.qlearner_manager = qlearner_manager
        self.mimic_manager = mimic_manager
        self.game_model_manager = game_model_manager

        self.custom_opponent = custom_opponent

        self.r = 0
    
    agent_name = cast(str, editable_dpg_value("agent_name"))
    reaction_time = cast(str, editable_dpg_value("reaction_time"))

    def _save_agent(self, s, a, u):
        folder_path = path.join("runs", "analysis_" + self.agent_name)
        self.agent.save(folder_path)

    def add_custom_elements(self, analyser: Analyser):
        with dpg.group(horizontal=True):
            dpg.add_text("Predicted opponent action:")
            dpg.add_text("X", tag="predicted_opponent_action")

        react = self.the_one.reaction_time_emulator
        if react is not None:
            with dpg.group(horizontal=True):
                dpg.add_text("Reaction time:")
                dpg.add_slider_int(min_value=react.min_time, max_value=react.max_time, width=200, tag="reaction_time", enabled=False)

        dpg.add_checkbox(label="Online learning", default_value=False, tag="the_one_online_learning")

        with dpg.group(horizontal=True):
            dpg.add_text("Name:")
            dpg.add_input_text(multiline=False, no_spaces=True, tag="agent_name", enabled=True)
            dpg.add_button(label="Save", callback=self._save_agent)

        dpg.add_separator()

        self.qlearner_manager.add_custom_elements(analyser)

        dpg.add_separator()

        if self.mimic_manager is not None:
            self.mimic_manager.add_custom_elements(analyser)
        
        dpg.add_separator()

        if self.game_model_manager is not None:
            self.game_model_manager.add_custom_elements(analyser)

    def on_state_update(self, analyser: Analyser):
        self.qlearner_manager.on_state_update(analyser)
        if self.mimic_manager is not None:
            self.mimic_manager.on_state_update(analyser)
        if self.game_model_manager is not None:
            self.game_model_manager.on_state_update(analyser)

        react = self.the_one.reaction_time_emulator
        if react is not None and react.previous_reaction_time is not None:
            self.reaction_time = react.previous_reaction_time

        if dpg.get_value("the_one_online_learning") and analyser.most_recent_transition is not None and not analyser.use_custom_action:
            obs, next_obs, reward, terminated, truncated, info, next_info = analyser.most_recent_transition.as_tuple()
            self.r += reward
            
            # Ignore hitstop/freeze
            in_hitstop = ActionMap.is_in_hitstop_ori(next_info, True) or ActionMap.is_in_hitstop_ori(next_info, False)
            if not (in_hitstop and obs.isclose(next_obs).all()):
                if isinstance(self.custom_opponent, CurriculumOpponent):
                    next_info["next_opponent_policy"] = self.custom_opponent.peek(next_info)
                self.agent.update(obs, next_obs, self.r, terminated, truncated, info, next_info)
                self.r = 0
    
    @property
    def the_one(self) -> TheOneAgent:
        if isinstance(self.agent, TrainingLoggerWrapper):
            return self.agent.agent
        return self.agent


def main(
    custom: bool = False,
    model: str = "to",
    load: str | None = None,
    log: str | None = None,
    opponent: Literal["human", "bot", "self"] | str = "bot",
    fast_forward: bool = True,
    include_gm: bool = False,
    dense_reward: bool = False,
    blank: bool = False,
):
    if not custom and not load:
        raise ValueError("should either use a custom agent or load one")
    
    agent_name: str = load if load else ""

    setup_logger("analyse", stdout_level=logging.DEBUG, log_to_file=False)

    env_kwargs = {}
    custom_opponent: Opponent | None = None
    if opponent == "human":
        env_kwargs["vs_player"] = True
    elif opponent == "bot":
        pass
    elif opponent == "self":
        env_kwargs["opponent"] = lambda o, i: (False, False, False) # just set a dummy
    else:
        custom_opponent = getattr(curriculum, opponent)()
        assert custom_opponent is not None
        env_kwargs["opponent"] = custom_opponent.act
    env_kwargs.update(FootsiesEnv.find_ports(15000))

    footsies_env = FootsiesEnv(
        game_path="../Footsies-Gym/Build/FOOTSIES.x86_64",
        render_mode="human",
        sync_mode="synced_non_blocking",
        fast_forward=opponent != "human" and fast_forward,
        dense_reward=dense_reward,
        log_file="out.log",
        log_file_overwrite=True,
        **env_kwargs,
    )

    env = TransformObservation(
        FootsiesSimpleActions(
            FlattenObservation(
                FootsiesNormalized(
                    footsies_env,
                )
            ),
            agent_allow_special_moves=False,
        ),
        lambda o: torch.from_numpy(o).float().unsqueeze(0),
    )
    assert env.observation_space.shape
    assert isinstance(env.action_space, Discrete)

    if custom:
        agent, loggables = to_(
            env.observation_space.shape[0],
            int(env.action_space.n),
        )
    
    else:
        parameters = load_agent_parameters(agent_name)
        agent, loggables = import_agent(model, env, parameters)
        agent = cast(TheOneAgent, agent)

    if log:
        logged_agent = TrainingLoggerWrapper(
            agent,
            10000,
            log_dir="runs/analysis",
            episode_reward=True,
            average_reward=True,
            win_rate=True,
            truncation=True,
            episode_length=True,
            test_states_number=1,
            **loggables, # type: ignore
        )

        logged_agent.preprocess(env)
    
    else:
        logged_agent = agent

    if agent.opp is not None and agent.opp.p2_model is not None and agent.opp.p2_model.network.is_recurrent:
        warnings.warn("Since the agent is recurrent, it needs to have 'online learning' enabled in order to have the recurrent state updated. Additionally, loading and saving states is discouraged for the same reason.")

    if load and not blank:
        load_agent(agent, agent_name)

    if opponent == "self":
        custom_opponent = agent.extract_opponent(env)
        footsies_env.set_opponent(custom_opponent.act)

    qlearner_manager = QLearnerAnalyserManager(
        agent.a2c,
        action_dim=agent.action_dim,
        opponent_action_dim=agent.opponent_action_dim,
        include_online_learning=False,
    )

    if agent.opp is not None:
        mimic_manager = MimicAnalyserManager(
            p1_model=None,
            p2_model=agent.opp.p2_model,
            p1_mirror_p2=False,
            include_online_learning=False,
        )
    
    else:
        mimic_manager = None

    if agent.gm is not None and include_gm:
        game_model_manager = GameModelAnalyserManager(
            agent=agent.gm
        )
    
    else:
        game_model_manager = None

    def act(obs, info):
        if isinstance(custom_opponent, CurriculumOpponent):
            info["next_opponent_policy"] = custom_opponent.peek(info)

        action = logged_agent.act(obs, info)

        if agent.recently_predicted_opponent_action is not None:
            predicted_opponent_action = ActionMap.simple_as_move(agent.recently_predicted_opponent_action)
        else:
            predicted_opponent_action = None
        dpg.set_value("predicted_opponent_action", predicted_opponent_action.name if predicted_opponent_action is not None else "X")

        return action

    manager = TheOneAnalyserManager(
        agent=logged_agent,
        qlearner_manager=qlearner_manager,
        mimic_manager=mimic_manager,
        game_model_manager=game_model_manager,
        custom_opponent=custom_opponent,
    )

    analyser = Analyser(
        env=env,
        p1_action_source=act,
        custom_elements_callback=manager.add_custom_elements,
        custom_state_update_callback=manager.on_state_update,
    )
    analyser.start(state_change_apply_immediately=True)


if __name__ == "__main__":
    tyro.cli(main)
