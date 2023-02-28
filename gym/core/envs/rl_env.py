import random
from pprint import pprint

import gymnasium as gym
import numpy as np
from core.envs.validator import Validator

ALPHA_INCREMENT_LIMIT = 0.3
ALPHA_DECREMENT_LIMIT = 0.3


class Environment(gym.Env):
    def __init__(self, num_validators=100, honest_ratio=0.5, initial_alpha=1, rounds=100, *args, **kwargs):
        """
        Initialize the environment.

        Parameters
        ----------
        num_validators : int
            number of validators
        honest_ratio : float
            ratio of honest validators
        initial_alpha : float
            initial alpha
        """
        self.num_validators = num_validators
        self.honest_ratio = honest_ratio
        self.initial_alpha = initial_alpha
        self.rounds = rounds
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
                self.validators.append(
                    Validator(initial_strategy='honest', id=i))
            else:
                self.validators.append(
                    Validator(initial_strategy='malicious', id=i))

        # shuffle the validators
        random.shuffle(self.validators)

        # define the action space
        self.action_space = gym.spaces.Box(
            low=-ALPHA_DECREMENT_LIMIT, high=ALPHA_INCREMENT_LIMIT, shape=(1,))

        # define the observation space.
        # Observation space is constructed by the following:
        # - sum of balance
        # - sum of effective balance
        # - proportion of honest validators
        self.observation_space = gym.spaces.Dict({
            "sum_of_balance": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
            "sum_of_effective_balance": gym.spaces.Box(low=0, high=np.inf, shape=(1,)),
            "honest_proportion": gym.spaces.Box(low=0, high=1, shape=(1,)),
        })

        self.alpha = initial_alpha
        self.rounds = rounds
        self.counter = 0

        super(Environment, self).__init__(*args, **kwargs)

    def _get_obs(self):
        """
        Get the observation of the environment.

        Returns
        -------
        payload : dict
            observation of the environment
        """
        payload = dict(
            sum_of_balance=self._get_sum_active_balance(),
            sum_of_effective_balance=self._get_sum_active_balance(),
            honest_proportion=self._get_honest_proportion()
        )
        return payload

    def _get_info(self):
        return {
            'round': self.counter,
            'alpha': self.alpha,
            'honest_proportion': self._get_honest_proportion(),
        }

    def _get_reward(self):
        return sum(
            [validator.get_balance()
             for validator in self.validators if validator.get_strategy() == 'honest']
        ) / sum(
            [validator.get_balance() for validator in self.validators]
        )

    def get_validator_info(self, verbose=False):
        """
        Get the information of the validators.

        Parameters
        ----------
        verbose : bool
            whether to print the information
        """
        payload = []
        for validator in self.validators:
            payload.append(dict(
                id=validator.id,
                strategy=validator.get_strategy(),
                balance=validator.get_balance(),
                effective_balance=validator.get_effective_balance()
            ))
        if verbose is True:
            pprint(payload)
        return payload

    def _get_sum_active_balance(self):
        """
        Get the sum of active balance of all validators.

        Returns
        -------
        sum : int
            sum of active balance of all validators
        """
        return sum(
            [validator.get_balance() for validator in self.validators]
        )

    def _get_honest_proportion(self):
        # return the proportion of honest validators
        return sum(
            [validator.get_strategy() == 'honest' for validator in self.validators]
        ) / len(self.validators)

    def reset(self):
        """
        Reset the environment and return an initial observation.

        Returns
        -------
        observation : numpy array
            The initial observation of the environment.
        """
        print('[INFO] Resetting the environment...')
        self.validators = []
        for i in range(self.num_validators):
            if i < self.num_validators * self.honest_ratio:
                self.validators.append(
                    Validator(initial_strategy='honest', id=i))
            else:
                self.validators.append(
                    Validator(initial_strategy='malicious', id=i))

        # shuffle the validators
        random.shuffle(self.validators)

        self.alpha = self.initial_alpha
        self.counter = 0

        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        Take a step in the environment.
        1. Select a validator as proposer (from honest validator)
        2. Proposing
        3. Voting
        4. Update Strategy
        4. Next Round

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
        # update alpha
        self.alpha += action[0]

        # select proposer from honest validators
        honest_validators = [
            validator for validator in self.validators if validator.get_strategy() == 'honest']
        proposer = random.choice(honest_validators)
        proposer_id = proposer.id

        # propose the block
        # get_base_reward(self, sum_of_active_balance)
        # propose(self, base_reward, honest_proportion)
        base_reward = proposer.get_base_reward(
            sum_of_active_balance=self._get_sum_active_balance())
        proposer.propose(base_reward, self._get_honest_proportion())

        # vote
        # validators except proposer votes
        # vote(self, base_reward, honest_proportion, alpha)
        for validator in self.validators:
            if validator.id != proposer_id:
                validator.vote(
                    base_reward=base_reward, honest_proportion=self._get_honest_proportion(), alpha=self.alpha)

        # update strategy
        for validator in self.validators:
            validator.update_strategy()

        # termination condition
        self.counter += 1
        if self.counter >= self.rounds:
            done = True
        else:
            done = False

        return self._get_obs(), self._get_reward(), done, self._get_info()

    def render(self, mode='human'):
        """
        Render the environment.

        Parameters
        ----------
        mode : str
            The mode to render the environment in.
        """
        print(
            f'[EPOCH {self.counter}] alpha: {self.alpha:.2f}, honest_proportion: {self._get_honest_proportion():.2f}, reward: {self._get_reward():.2f}, sum_of_balance: {self._get_sum_active_balance():.2f}, sum_of_effective_balance: {self._get_sum_active_balance():.2f}')

    def close(self):
        """
        Clean up any resources used by the environment.
        """
        raise NotImplementedError
