class CreditAssignment:
    def __init__(self, lambda_decay=0.9, eta=0.01):
        self.lambda_decay = lambda_decay
        self.eta = eta
        self.eligibility_store = {}
        self.recent_rewards = []

    def update_traces(self, population):
        population.update_eligibility_traces()

    def apply_credit(self, population, reward_delta):
        self.recent_rewards.append({
            'delta': reward_delta,
            'magnitude': abs(reward_delta)
        })

        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)

        population.apply_credit(reward_delta, self.eta)
        population.decay_eligibility_traces(self.lambda_decay)

    def get_accumulated_reward(self, window=100):
        recent = self.recent_rewards[-window:] if len(self.recent_rewards) > window else self.recent_rewards
        if not recent:
            return 0.0
        return sum(r['delta'] for r in recent) / len(recent)
