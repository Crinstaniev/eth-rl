class Validator(object):
    def __init__(self, initial_strategy) -> None:
        if initial_strategy not in ['honest', 'malicious']:
            raise ValueError('initial strategy should be either honest or malicious')
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
    
    def reward(self):
        # reward the validator
        pass 
    
    def penalize(self):
        # penalize the validator
        pass