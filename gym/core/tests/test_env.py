import numpy as np
from core.envs.rl_env import Environment
from stable_baselines3 import A2C, DDPG, DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.envs import SimpleMultiObsEnv
from stable_baselines3.common.noise import (NormalActionNoise,
                                            OrnsteinUhlenbeckActionNoise)
from tqdm import tqdm

import gym
from gym.utils.env_checker import check_env

HONEST_RATIO = 0.8
EPOCHS = 30


def test_api_conformity():
    env = Environment(num_validators=500, honest_ratio=HONEST_RATIO)
    check_env(env, warn=True)


def test_rl():
    env = Environment(num_validators=256,
                      honest_ratio=HONEST_RATIO, rounds=64)

    model = A2C("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000, progress_bar=True)
    model.save("a2c")
