from footsies_gym.envs.footsies import FootsiesEnv
import torch
from tests.conftest import wrap_env_standard
from models import to_
from opponents.curriculum import WhiffPunisher


def test_standard(footsies_env: FootsiesEnv):
    env = wrap_env_standard(footsies_env)

    agent, _ = to_(
        observation_space_size=env.observation_space.shape[0],
        action_space_size=env.action_space.n,
        use_opponent_model=True,
    )

    obs, info = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        action = agent.act(obs, info)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        agent.update(obs, next_obs, reward, terminated, truncated, info, next_info)


def test_whiff_punisher(footsies_env: FootsiesEnv):
    env = wrap_env_standard(footsies_env)

    agent, _ = to_(
        observation_space_size=env.observation_space.shape[0],
        action_space_size=env.action_space.n,
    )

    whiff_punisher = WhiffPunisher()
    footsies_env.set_opponent(whiff_punisher.act)

    obs, info = env.reset()
    terminated, truncated = False, False

    next_opponent_policy = whiff_punisher.peek(info)
    info["next_opponent_policy"] = next_opponent_policy

    while not (terminated or truncated):        
        action = agent.act(obs, info)
        next_obs, reward, terminated, truncated, next_info = env.step(action)

        assert agent.recently_predicted_opponent_action == torch.argmax(next_opponent_policy).item()
        
        next_opponent_policy = whiff_punisher.peek(next_info)
        next_info["next_opponent_policy"] = next_opponent_policy

        agent.update(obs, next_obs, reward, terminated, truncated, info, next_info)
        
        obs = next_obs
        info = next_info
