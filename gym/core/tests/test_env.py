from core.envs.rl_env import Environment

def test_env_init():
    env = Environment(num_validators=500, honest_ratio=0.7)
    return env 

def test_get_active_balance():
    env = test_env_init()
    active_balance = env._get_sum_active_balance()
    print(active_balance)
    return

def test_get_validators_info():
    env = test_env_init()
    validators_info = env.get_validator_info(verbose=True)
    