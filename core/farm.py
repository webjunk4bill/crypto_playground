class Farm:
    def __init__(self, token_a, token_b, weekly_reward, tvl_reward, **kwargs):
        self.low = None
        self.high = None
        self.token_a = token_a
        self.token_b = token_b
        self.weekly_reward = weekly_reward
        self.tvl_reward = tvl_reward
        self.tick_spacing = kwargs.get('tick_spacing', 100)
        self._seed = None

    def set_seed(self, seed):
        self._seed = seed

    @property
    def seed(self):
        return self._seed

    @property
    def apr_pt(self):
        return self.weekly_reward / self.tvl_reward * 52 * 100

    @property
    def fee_ph_pt(self):
        return self.apr / 100 / 365 / 24 * self.seed

    def set_range(self, low, high):
        self.low = int(low)
        self.high = int(high)
        high_tick = int(self.high * self.tick_spacing / 100)
        low_tick = int(self.low * self.tick_spacing / 100)
        self.fee_ph = self.apr_pt / 100 / 365 / 24 * self.seed


    #todo: clean up, Bill already has this, should really read all the code before creating new classes.... Convert to model class.
