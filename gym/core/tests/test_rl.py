import numpy as np
import pytest
from core.envs.rl_env import Environment
from stable_baselines3 import A2C, DDPG
from stable_baselines3.common.noise import NormalActionNoise

EPOCHS = 30


def test_a2c():
    env = Environment(num_validators=256,
                      honest_ratio=0.5, rounds=32)

    model = A2C("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=32 * 50, progress_bar=True)
    model.save("models/a2c")


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
