import collections
import math

import numpy as np
from core.utils.helper_functions import deprecated

BASE_REWARD_FACTOR = 64
# BASE_REWARD_FACTOR = 1
BASE_REWARD_PER_EPOCH = 4
STRATEGY_UPDATE_BALANCE_FACTOR = 32


class Validator(object):
    """
    Validator class

    Parameters
    ----------
    initial_strategy : str
        initial strategy of the validator
    id : int
        id of the validator

    Attributes
    ----------
    id : int
        id of the validator
    strategy : str
        strategy of the validator
    balance : float
        balance of the validator
    effective_balance : float
        effective balance of the validator
    mutable : bool
        whether the strategy of the validator is mutable
    """

    def __init__(self, initial_strategy, id) -> None:
        """
        If the validator is honest:
        - When proposing, it will propose the block on time
        - When voting, it will vote
        If the validator is malicious:
        - When proposing, it will propose
        - When voting, it will not vote
        """
        if initial_strategy not in ['honest', 'malicious']:
            raise ValueError(
                'initial strategy should be either honest or malicious')
        self.id = id
        self.strategy = initial_strategy
        self.balance = 32
        self.effective_balance = 32

        if self.strategy == 'honest':
            self.mutable = True
        else:
            # self.mutable = False
            self.mutable = True

        # initialize strategy update-specific variables
        self.balance_history = collections.deque([self.balance], maxlen=10)
        self.balance_decreasing_count = 0

    def get_effective_balance(self):
        """
        Get the effective balance of the validator

        Returns
        -------
        effective_balance : float
            effective balance of the validator
        """
        # limit: 32 Ether
        # Calculation: current balance decrease 0.5, effective balance decrease 1; Current balance increase 1.25, effective balance increase 1
        return max(0, self.effective_balance)

    def get_balance(self):
        """
        Get the balance of the validator

        Returns
        -------
        balance : float
            balance of the validator
        """
        return max(0, self.balance)

    def get_strategy(self):
        """
        Get the strategy of the validator

        Returns
        -------
        strategy : str
            strategy of the validator
        """
        return self.strategy

    def get_base_reward(self, sum_of_active_balance):
        """
        Get the base reward of the validator

        Parameters
        ----------
        sum_of_active_balance : float
            sum of effective balance of all active validators

        Returns
        -------
        base_reward : float
            base reward of the validator
        """
        effective_balance = self.get_effective_balance()
        # temporary solution
        sum_of_active_balance = max(1, sum_of_active_balance)
        base_reward = (
            effective_balance * (
                BASE_REWARD_FACTOR /
                (BASE_REWARD_PER_EPOCH * math.sqrt(sum_of_active_balance))
            )
        )
        # print('base_reward: ', base_reward)
        return max(0, base_reward)

    def increase_balance(self, amount):
        """
        decrease balance

        Parameters
        ----------
        amount : float
            amount to increase
        """
        self.balance += amount
        # 1 effective balance increase per 1.25 balance increase if balance increament larger than 1.25
        if amount >= 1.25:
            self.effective_balance += math.floor(amount / 1.25)

        return

    def decrease_balance(self, amount):
        """
        decrease balance

        Parameters
        ----------
        amount : float
            amount to decrease
        """
        self.balance -= amount
        if amount >= 0.5:
            self.effective_balance -= math.floor(amount / 0.5)
        return

    def propose(self, base_reward, honest_proportion):
        """
        Validator propose a block.
        If the validator is honest, it will propose the block on time. So it will receive reward.
        If the validator is malicious, it will not propose a block. But it will not be panalized.

        Parameters
        ----------
        base_reward : float
            base reward
        honest_proportion : float
            proportion of honest validators
        """
        if self.strategy == 'honest':
            # get reward
            proposing_reward = (
                (1 / 8) * base_reward * honest_proportion
            )
            self.increase_balance(proposing_reward)
        self._sync_committee_reward(
            base_reward=base_reward, honest_proportion=honest_proportion)
        return

    def vote(self, base_reward, honest_proportion, alpha):
        """
        Validator vote for a block.
        If the validator is honest, it will vote. So it will receive reward.
        If the validator is malicious, it will not vote. So it will be penalized.

        Parameters
        ----------
        base_reward : float
            base reward
        honest_proportion : float
            proportion of honest validators
        alpha : float
            penalization factor
        """
        voting_reward = (
            (27 / 32) * base_reward * honest_proportion
        )
        if self.strategy == 'honest':
            # get reward
            self.increase_balance(voting_reward)
        else:
            # get penalized
            self.decrease_balance(alpha * voting_reward)
        self._sync_committee_reward(
            base_reward=base_reward, honest_proportion=honest_proportion)
        return

    def _sync_committee_reward(self, base_reward, honest_proportion):
        """
        Validator receive reward for participating in sync committee.

        Parameters
        ----------
        base_reward : float
            base reward
        honest_proportion : float
            proportion of honest validators
        """
        sync_committee_reward = (
            (1 / 32) * base_reward * honest_proportion
        )
        self.increase_balance(sync_committee_reward)
        return

    @deprecated
    def update_strategy_old(self):
        """
        Update the strategy of the validator. If the validator is honest, the probability it became malicious is negatively related to its balance. If the validator is malicious and it is mutable, the probability it became honest is positively related to its balance. The probability follows a logistic function.
        """
        if self.strategy == 'honest':
            # with less balance, more possible to become malicious
            probability_malicious = 1 / \
                (1 + np.exp(self.balance - STRATEGY_UPDATE_BALANCE_FACTOR))
            # print('probability_malicious', probability_malicious)
            if np.random.random() < probability_malicious:
                self.strategy = 'malicious'
        else:
            if self.mutable:
                # more balance, more possible to become honest
                probability_honest = 1 / \
                    (1 + np.exp(STRATEGY_UPDATE_BALANCE_FACTOR - self.balance))
                # print('probability_honest', probability_honest)
                if np.random.random() < probability_honest:
                    self.strategy = 'honest'
        return

    @deprecated
    def update_strategy_old(self):
        """
        When a honest validator founds its rewards decreasing for a long time, it will become malicious. When a malicious validator founds its rewards decreasing for a long time, it will also become honest.
        """
        # Next: linear->softmax
        if self.strategy == 'honest':
            if self.balance < self.balance_history[-1]:
                self.balance_decreasing_count += 1
            else:
                self.balance_decreasing_count = 0
            if self.balance_decreasing_count >= 5:
                self.strategy = 'malicious'
                # print('honest->malicious')
        else:
            if self.mutable:
                if self.balance < self.balance_history[-1]:
                    self.balance_decreasing_count += 1
                else:
                    self.balance_decreasing_count = 0
                if self.balance_decreasing_count >= 5:
                    self.strategy = 'honest'
                    # print('malicious->honest')

        self.balance_history.append(self.balance)
        return

    def update_strategy(self, new_strategy):
        """
        Revolutional strategy to update validator behavior.
        """
        self.strategy = new_strategy
        return
