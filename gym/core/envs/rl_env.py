import random
from pprint import pprint

import gymnasium as gym
from core.envs.validator import Validator


class Environment(gym.Env):
    def __init__(self, num_validators=100, honest_ratio=0.5, *args, **kwargs):
        
        # assert num_validator is integer and greater than 0
        if not isinstance(num_validators, int) or num_validators <= 0:
            raise ValueError('num_validators should be a positive integer')
        
        # assert honest_ratio is float between 0 and 1
        if not isinstance(honest_ratio, float) or honest_ratio < 0 or honest_ratio > 1:
            raise ValueError('honest_ratio should be a float between 0 and 1')
        
        # initialize validators and their strategies
        self.validators = []
        for i in range(num_validators):
            if i < num_validators * honest_ratio:
                self.validators.append(Validator(initial_strategy='honest'))
            else:
                self.validators.append(Validator(initial_strategy='malicious'))
        
        # shuffle the validators
        random.shuffle(self.validators)
        
        super(Environment, self).__init__(*args, **kwargs)
        
    def get_validator_info(self, verbose=False):
        payload = []
        for validator in self.validators:
            payload.append(dict(
                strategy=validator.get_strategy(),
                balance=validator.get_balance(),
                effective_balance=validator.get_effective_balance()
            ))
        if verbose is True:
            pprint(payload)
        return payload
    
    def _get_sum_active_balance(self):
        # return the sum of everyone's balance
        return sum(
            [validator.get_balance() for validator in self.validators]
        )


    def reset(self):
        """
        Reset the environment and return an initial observation.

        Returns
        -------
        observation : numpy array
            The initial observation of the environment.
        """
        raise NotImplementedError

    def step(self, action):
        """
        Take a step in the environment.

        Parameters
        ----------
        action : int
            The action to take in the environment.

        Returns
        -------
        observation : numpy array
            The new observation of the environment after taking the action.
        reward : float
            The reward obtained after taking the action.
        done : bool
            Whether the episode has ended or not.
        info : dict
            Additional information about the step.
        """
        raise NotImplementedError

    def render(self, mode='human'):
        """
        Render the environment.

        Parameters
        ----------
        mode : str
            The mode to render the environment in.
        """
        raise NotImplementedError

    def close(self):
        """
        Clean up any resources used by the environment.
        """
        raise NotImplementedError
