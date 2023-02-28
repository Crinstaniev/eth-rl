import math


BASE_REWARD_FACTOR = 64
BASE_REWARD_PER_EPOCH = 4


class Validator(object):

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

    def get_effective_balance(self):
        # limit: 32 Ether
        # Calculation: current balance decrease 0.5, effective balance decrease 1; Current balance increase 1.25, effective balance increase 1
        return self.effective_balance

    def get_balance(self):
        return self.balance

    def get_strategy(self):
        return self.strategy

    def get_base_reward(self, sum_of_active_balance):
        effective_balance = self.get_effective_balance()
        base_reward = (
            effective_balance * (
                BASE_REWARD_FACTOR /
                (BASE_REWARD_PER_EPOCH * math.sqrt(sum_of_active_balance))
            )
        )
        return base_reward

    def increase_balance(self, amount):
        """
        decrease balance

        Parameters
        ----------
        amount : float
            amount to increase
        """
        self.balance += amount
        self.effective_balance += amount / 1.25
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
        self.effective_balance -= amount / 0.5
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
