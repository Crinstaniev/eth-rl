from core.envs.rl_env import Environment

from gym.utils.env_checker import check_env


def test_api_conformity():
    env = Environment(num_validators=500, honest_ratio=0.8)
    check_env(env, warn=True)
