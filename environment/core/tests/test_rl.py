import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from core.envs.rl_env import Environment
from core.utils.helper_functions import SaveOnBestTrainingRewardCallback
from stable_baselines3 import A2C, DDPG
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.results_plotter import plot_results

EPOCHS = 30
ROUNDS = 64
NUM_VALIDATORS = 512
HONEST_RATIO = 0.5

np.random.seed(42)


def test_stable_alpha():
    env = Environment(num_validators=NUM_VALIDATORS,
                      honest_ratio=HONEST_RATIO, rounds=ROUNDS)
    observation = env.reset()

    info_records = []

    for _ in range(ROUNDS):
        action = [1]
        observation, reward, done, info = env.step(action)
        info_records.append(info)
        if done:
            observation = env.reset()

    df = pd.DataFrame(info_records)
    df.to_csv("results/stable_alpha.csv", index=False)


# @pytest.mark.skip()
def test_a2c():
    env = Environment(num_validators=NUM_VALIDATORS,
                      honest_ratio=HONEST_RATIO, rounds=ROUNDS)

    # Create log dir
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Create and wrap the environment
    env = Monitor(env, log_dir)

    timesteps = ROUNDS * 32

    model = A2C("MultiInputPolicy", env, verbose=1)
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=timesteps,
                progress_bar=True, callback=callback)
    model.save("models/a2c")

    info_records = []

    observation = env.reset()
    for _ in range(ROUNDS):
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        info_records.append(info)
        if done:
            observation = env.reset()

    df = pd.DataFrame(info_records)
    df.to_csv("results/a2c.csv", index=False)


# @pytest.mark.skip(reason="too slow without gpu")
def test_ddpg():
    env = Environment(num_validators=NUM_VALIDATORS,
                      honest_ratio=0.5, rounds=ROUNDS)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(
        n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=ROUNDS * 32, log_interval=10, progress_bar=True)
    model.save("models/ddpg")

    observation = env.reset()
    info_records = []
    for _ in range(ROUNDS):
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        info_records.append(info)
        if done:
            observation = env.reset()

    df = pd.DataFrame(info_records)
    df.to_csv("results/ddpg.csv", index=False)
