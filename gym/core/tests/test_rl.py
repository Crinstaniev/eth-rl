import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from core.envs.rl_env import Environment
from stable_baselines3 import A2C, DDPG
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.results_plotter import plot_results
from core.utils.helper_functions import SaveOnBestTrainingRewardCallback

EPOCHS = 30


def test_a2c():
    env = Environment(num_validators=256,
                      honest_ratio=0.5, rounds=32)

    # Create log dir
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Create and wrap the environment
    env = Monitor(env, log_dir)

    timesteps = 32 * (2 ** 8)

    model = A2C("MultiInputPolicy", env, verbose=1)
    callback = SaveOnBestTrainingRewardCallback(
        check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=timesteps,
                progress_bar=True, callback=callback)
    model.save("models/a2c")

    plot_results([log_dir], timesteps,
                 results_plotter.X_TIMESTEPS, "A2C")
    plt.savefig("results/a2c.png")
    plt.show()


@pytest.mark.skip(reason="too slow without gpu")
def test_ddpg():
    env = Environment(num_validators=256,
                      honest_ratio=0.5, rounds=16)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(
        n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MultiInputPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=10000, log_interval=10)
    model.save("models/ddpg")
