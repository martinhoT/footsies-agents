import torch
import dearpygui.dearpygui as dpg
import numpy as np
from gymnasium.wrappers.flatten_observation import FlattenObservation
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers.action_comb_disc import FootsiesActionCombinationsDiscretized
from footsies_gym.wrappers.normalization import FootsiesNormalized
from footsies_gym.moves import FootsiesMove
from scripts.analysis.analysis import Analyser
from agents.action import ActionMap
from main import load_agent
from scripts.analysis.analyse_a2c_qlearner import QLearnerAnalyserManager
from scripts.analysis.analyse_opponent_model import MimicAnalyserManager
from models import to_
from gymnasium.wrappers.transform_observation import TransformObservation
from main import setup_logger, load_agent, load_agent_parameters, import_agent
from opponents.curriculum import WhiffPunisher, Backer, UnsafePunisher, NSpammer, CurriculumOpponent
from agents.wrappers import FootsiesPhasicMoveProgress, FootsiesSimpleActions
from agents.logger import TrainingLoggerWrapper
import logging
import warnings


CUSTOM = False
MODEL = "to"
NAME = "f_opp"
LOAD = True
LOG = False
ROLLBACK_IF_POSSIBLE = True

if __name__ == "__main__":

    setup_logger("analyse", stdout_level=logging.DEBUG, log_to_file=False)

    custom_opponent = WhiffPunisher()

    footsies_env = FootsiesEnv(
        game_path="../Footsies-Gym/Build/FOOTSIES.x86_64",
        game_port=15000,
        opponent_port=15001,
        remote_control_port=15002,
        render_mode="human",
        sync_mode="synced_non_blocking",
        fast_forward=False,
        dense_reward=False,
        # vs_player=True,
        # opponent=custom_opponent.act,
    )

    env = TransformObservation(
        FootsiesSimpleActions(
            FlattenObservation(
                # FootsiesPhasicMoveProgress(
                    FootsiesNormalized(
                        footsies_env,
                        # normalize_guard=False,
                    )
                # )
            )
        ),
        lambda o: torch.from_numpy(o).float().unsqueeze(0),
    )

    if CUSTOM:
        agent, loggables = to_(
            env.observation_space.shape[0],
            env.action_space.n,

            actor_entropy_coef=0.0,
            critic_agent_update="expected_sarsa",
            critic_opponent_update="expected_sarsa",
            maxent=0.01,
            maxent_gradient_flow=True,
            use_opponent_model=True,
        )
    
    else:
        parameters = load_agent_parameters(NAME)
        parameters["rollback"] = ROLLBACK_IF_POSSIBLE and not parameters.get("use_opponent_model", False)
        agent, loggables = import_agent(MODEL, env, parameters)

    if LOG:
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
            **loggables,
        )

        logged_agent.preprocess(env)
    
    else:
        logged_agent = agent

    # idle_distribution = torch.tensor([0.0] * ActionMap.n_simple()).float().unsqueeze(0)
    # idle_distribution[0, 0] = 1.0

    if agent.opp.p2_model.network.is_recurrent:
        warnings.warn("Since the agent is recurrent, it needs to have 'online learning' enabled in order to have the recurrent state updated. Additionally, loading and saving states is discouraged for the same reason.")

    if LOAD:
        load_agent(agent, NAME)

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

    def custom_elements_callback(analyser: Analyser):
        with dpg.group(horizontal=True):
            dpg.add_text("Predicted opponent action:")
            dpg.add_text("X", tag="predicted_opponent_action")
        
        dpg.add_checkbox(label="Online learning", default_value=False, tag="the_one_online_learning")

        qlearner_manager.add_custom_elements(analyser)

        dpg.add_separator()

        if mimic_manager is not None:
            mimic_manager.include_mimic_dpg_elements(analyser)

    def custom_state_update_callback(analyser: Analyser):
        qlearner_manager.on_state_update(analyser)
        if mimic_manager is not None:
            mimic_manager.predict_next_move(analyser)
        
        if dpg.get_value("the_one_online_learning") and analyser.most_recent_transition is not None and not analyser.use_custom_action:
            obs, next_obs, reward, terminated, truncated, info, next_info = analyser.most_recent_transition
            if custom_opponent is not None:
                next_info["next_opponent_policy"] = custom_opponent.peek(next_info)
            logged_agent.update(obs, next_obs, reward, terminated, truncated, info, next_info)

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

    analyser = Analyser(
        env=env,
        # p1_action_source=lambda o, i: next(p1),
        p1_action_source=act,
        custom_elements_callback=custom_elements_callback,
        custom_state_update_callback=custom_state_update_callback,
    )
    analyser.start(state_change_apply_immediately=True)
