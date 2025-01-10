import unimath as um

class Token:
    def __init__(self, symbol, price):
        self.symbol = symbol
        self.price = price
        self.balance = 0
        
    @property
    def value(self):
        return self.balance * self.price


class LiquidityPool:
    def __init__(self, token_x: Token, token_y: Token):
        """
        Native price is y/x price or else liquidity calculations won't work
        token x and y need to be selected such that when the lower range is hit, it is all token x and the upper range is all token y
        """
        self.token_x = token_x
        self.token_y = token_y
        self.liquidity = None
        self.lower_range = None
        self.upper_range = None
        self.init_x_bal = None
        self.init_x_price = None
        self.init_y_bal = None
        self.init_y_price = None
        self.seed = None
        self.apr = 0
        self.initial_setup = True
        self.duration = 0
    
    @property
    def value(self):
        return self.token_x.value + self.token_y.value
    
    @property
    def native_price(self):
        """
        For the math to work out, we need how many of y per x, so if x = BTC and y = ETH, 
        the native price would be the USD value of BTC / the USD value of ETH, which would yield ETH per BTC
        Or, (x USD/BTC) / (y USD/ETH) = ETH/BTC (y/x)
        """
        return self.token_x.price / self.token_y.price
    
    @property
    def hold_value(self):
        """ 
        Determines the value if the original tokens were just held instead of put into a CLP 
        Need to deal with USDC being one of the tokens vs something like BTC/ETH LP
        """
        if self.seed:
            if self.token_x.symbol == 'USDC':
                return (self.seed / self.init_y_price) * self.token_y.price
            elif self.token_y.symbol == 'USDC':
                return (self.seed / self.init_x_price) * self.token_x.price
            else:
                return self.init_x_bal * self.token_x.price + self.init_y_bal * self.token_y.price
            
    @property
    def impermanent_loss(self):
        loss = self.hold_value - self.value
        if loss > 0:
            return self.hold_value - self.value 
        else:
            return 0
    
    def create_range(self, seed, lower_pct, upper_pct):
        # from a USD perspective, more money will be invested in token x if the ratio favors the upper range
        ratio = upper_pct / (upper_pct  + lower_pct)
        self.token_x.balance = seed * ratio / self.token_x.price
        self.token_y.balance = seed * (1 - ratio) / self.token_y.price
        self.upper_range = self.native_price * (1 + upper_pct/100)
        self.lower_range = self.native_price / (1 + lower_pct/100)
        # re-calc upper range to make liquidity come out even
        self.upper_range = um.calc_upper_range_pb(self.token_x.balance, self.token_y.balance, self.lower_range, self.native_price)
        # Calc token amounts based on desired ratio and liquidity
        liq_x = um.liquidity_x(self.token_x.balance, self.native_price, self.upper_range)
        liq_y = um.liquidity_y(self.token_y.balance, self.native_price, self.lower_range)
        self.liquidity = min(liq_x, liq_y)
        # Make note of the initial token values to look at impermanent loss
        if self.initial_setup:
            self.seed = seed
            self.init_x_price = self.token_x.price
            self.init_y_price = self.token_y.price
            self.init_x_bal = self.token_x.balance
            self.init_y_bal = self.token_y.balance
            self.initial_setup = False

    def update_token_balances(self, duration):
        """ 
        This should be run after token price changes are made 
        The duration is how long has elapsed between the initial deposit and the current price.  Used to calculate the APR.
        """
        self.duration += duration  # add duration days
        # Check to make sure not out of range
        if self.native_price < self.lower_range:
            price = self.lower_range
        elif self.native_price > self.upper_range:
            price = self.upper_range
        else:
            price = self.native_price
        self.token_x.balance = um.calc_amount_x(self.liquidity, price, self.upper_range)
        self.token_y.balance = um.calc_amount_y(self.liquidity, price, self.lower_range)
        self.calc_apr_for_duration()

    def rebalance(self, lower_pct, upper_pct):
        seed = self.value
        self.create_range(seed, lower_pct, upper_pct)

    def calc_apr_for_duration(self):
        if self.impermanent_loss > 0:
            self.apr = (self.impermanent_loss / self.value) * (365 / self.duration) * 100
        else:
            self.apr = 0

    def __repr__(self):
        return (
            f"{self.token_x.symbol}: {self.token_x.balance} | {self.token_y.symbol}: {self.token_y.balance}\n"
            f"LP Value: ${self.value:.2f}\n"
            f"Current Price: {self.native_price} in {self.token_y.symbol}/{self.token_x.symbol}\n"
            f"Range: {self.lower_range} ~ {self.upper_range}\n"
            f"Hold Value: ${self.hold_value:.2f}\n"
            f"Impermanent Loss: ${self.impermanent_loss:.2f}\n"
            f"Fee APR required to offset IL: {self.apr:.1f}%\n"
        )


