from core.envs.rl_env import Environment
from tqdm import tqdm
# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2


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

    for _ in range(150):
        # agent policy that uses the observation and info
        action = env.action_space.sample()
        observation, _, terminated, info = env.step(action)
        env.render()
        if terminated:
            observation, info = env.reset()


# def test_dqn():
#     env = Environment(num_validators=500, honest_ratio=0.7)

#     model = PPO2(MlpPolicy, env, verbose=1)
#     model.learn(total_timesteps=10000)

#     obs, _ = env.reset()
#     for i in range(1000):
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info = env.step(action)
#         env.render()
