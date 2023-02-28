from core.envs.rl_env import Environment
from tqdm import tqdm


def test_env_init():
    Environment(num_validators=500, honest_ratio=0.7)


def test_get_active_balance():
    env = Environment(num_validators=500, honest_ratio=0.7)
    active_balance = env._get_sum_active_balance()
    print(active_balance)


def test_get_validators_info():
    env = Environment(num_validators=500, honest_ratio=0.7)
    env.get_validator_info(verbose=False)


def test_sample_action():
    env = Environment(num_validators=500, honest_ratio=0.7)
    action = env.action_space.sample()
    print(action)


def test_get_obs():
    env = Environment(num_validators=500, honest_ratio=0.7)
    obs = env._get_obs()
    print(obs)


def test_get_reward():
    env = Environment(num_validators=500, honest_ratio=0.7)
    reward = env._get_reward()
    print(reward)


def test_step():
    env = Environment(num_validators=500, honest_ratio=0.7)
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(obs, reward, done, info)


def test_step_multiple_round():
    env = Environment(num_validators=500, honest_ratio=0.7)
    observation, info = env.reset()
    print('initial observation:', observation)
    print('initial info:', info)

    for _ in range(1000):
        # agent policy that uses the observation and info
        action = env.action_space.sample()
        observation, reward, terminated, info = env.step(action)
        print(info)
        if terminated:
            observation, info = env.reset()
