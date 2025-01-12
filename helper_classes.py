import unimath as um
import pandas as pd
import numpy as np
import requests
import time

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
        """
        Negative value means that holding the LP leads to "impermanent gains"
        """
        return self.hold_value - self.value 
        
    @property
    def ratio(self):
        return (self.native_price - self.lower_range) / (self.upper_range - self.lower_range)
        
    def get_price_ranges(self, lower_pct, upper_pct):
        # from a USD perspective, more money will be invested in token x if the ratio favors the upper range
        self.upper_range = self.native_price * (1 + upper_pct/100)
        self.lower_range = self.native_price / (1 + lower_pct/100)
        
    def initialize_range(self, seed, lower_pct, upper_pct):
        self.get_price_ranges(lower_pct, upper_pct)
        self.add_liquidity(seed)
        # Make note of the initial token values to look at impermanent loss
        if self.initial_setup:
            self.seed = seed
            self.init_x_price = self.token_x.price
            self.init_y_price = self.token_y.price
            self.init_x_bal = self.token_x.balance
            self.init_y_bal = self.token_y.balance
            self.initial_setup = False

    def add_liquidity(self, seed):
        # As price moves to the upper range (and therefore the "radio"), the amount of x is decreasing and the amount of y is increasing
        self.token_x.balance = seed * (1 - self.ratio) / self.token_x.price
        self.token_y.balance = seed * self.ratio / self.token_y.price
        # re-calc upper range to make liquidity come out even
        self.upper_range = um.calc_upper_range_pb(self.token_x.balance, self.token_y.balance, self.lower_range, self.native_price)
        # Calculate liquidity
        liq_x = um.liquidity_x(self.token_x.balance, self.native_price, self.upper_range)
        liq_y = um.liquidity_y(self.token_y.balance, self.native_price, self.lower_range)
        self.liquidity = min(liq_x, liq_y)
        
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
        self.get_price_ranges(lower_pct, upper_pct)
        self.add_liquidity(seed)

    def calc_apr_for_duration(self):
        if self.impermanent_loss > 0:
            self.apr = (self.impermanent_loss / self.value) * (365 / self.duration) * 100
        else:
            self.apr = 0

    def coumpound_fees(self, fees):
        """
        Doing this will slightly update the ranges given liquidity never coming out exact
        On a platfom like VFAT, you end up refunded a small amount of the compounded fees to make it round out
        """
        seed = self.value + fees
        self.add_liquidity(seed)
        self.calc_apr_for_duration()

    def __repr__(self):
        return (
            f"{self.token_x.symbol}: {self.token_x.balance:.6f} (${self.token_x.value:.2f})| {self.token_y.symbol}: {self.token_y.balance:.6f} (${self.token_y.value:.2f})\n" 
            f"LP Value: ${self.value:.2f}\n"
            f"Current Price: {self.native_price:.6f} in {self.token_y.symbol}/{self.token_x.symbol}\n"
            f"Range: {self.lower_range:.6f} ~ {self.upper_range:.6f}\n"
            f"Hold Value: ${self.hold_value:.2f}\n"
            f"Impermanent Loss: ${self.impermanent_loss:.2f}\n"
            f"Fee APR required to offset IL: {self.apr:.1f}%\n"
        )


def get_historical_prices(token_id, start_date: pd.Timestamp, end_date: pd.Timestamp, vs_currency='usd'):
    start_url = "https://api.coingecko.com/api/v3/coins"
    url = f"{start_url}/{token_id}/market_chart/range"

    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    params = {
        'vs_currency': vs_currency,
        'from': start_timestamp,
        'to': end_timestamp
    }

    print(f'Getting historical in_data for {token_id}')
    check = True
    increment = 1
    while check:
        if increment > 10:
            raise Exception("Tried 10 times with no valid response, aborting")
        print(f"Try #{increment}")
        try:
            response = requests.get(url, params=params)
            data = response.json()
            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            prices.set_index('timestamp', inplace=True)
            prices.index = pd.to_datetime(prices.index, unit='ms')
            prices['price'] = prices['price'].astype('float64')
            prices2 = prices['price']
            volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            volumes.set_index('timestamp', inplace=True)
            volumes.index = pd.to_datetime(volumes.index, unit='ms')
            volumes2 = volumes['volume']
            return [prices2, volumes2]
        except Exception as e:
            wait_sec = 15 * 2 ** increment
            print(f"Error fetching in_data due to {e}")
            print(f"Waiting {wait_sec} seconds and then will re-try")
            time.sleep(wait_sec)
            increment += 1


class Brownian():
    """
    A Brownian motion class constructor
    """
    def __init__(self,x0=0):
        """
        Init class
        """
        assert (type(x0)==float or type(x0)==int or x0 is None), "Expect a float or None for the initial value"
        
        self.x0 = float(x0)
    
    def gen_random_walk(self,n_step=100):
        """
        Generate motion by random walk
        
        Arguments:
            n_step: Number of steps
            
        Returns:
            A NumPy array with `n_steps` points
        """
        # Warning about the small number of steps
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")
        
        w = np.ones(n_step)*self.x0
        
        for i in range(1,n_step):
            # Sampling from the Normal distribution with probability 1/2
            yi = np.random.choice([1,-1])
            # Weiner process
            w[i] = w[i-1]+(yi/np.sqrt(n_step))
        
        return w
    
    def gen_normal(self,n_step=100):
        """
        Generate motion by drawing from the Normal distribution
        
        Arguments:
            n_step: Number of steps
            
        Returns:
            A NumPy array with `n_steps` points
        """
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")
        
        w = np.ones(n_step)*self.x0
        
        for i in range(1,n_step):
            # Sampling from the Normal distribution
            yi = np.random.normal()
            # Weiner process
            w[i] = w[i-1]+(yi/np.sqrt(n_step))
        
        return w
    
    def stock_price(
                    self,
                    s0=100,
                    mu=0.2,
                    sigma=0.68,
                    deltaT=52,
                    dt=0.1
                    ):
        """
        Models a stock price S(t) using the Weiner process W(t) as
        `S(t) = S(0).exp{(mu-(sigma^2/2).t)+sigma.W(t)}`
        
        Arguments:
            s0: Iniital stock price, default 100
            mu: 'Drift' of the stock (upwards or downwards), default 1
            sigma: 'Volatility' of the stock, default 1
            deltaT: The time period for which the future prices are computed, default 52 (as in 52 weeks)
            dt (optional): The granularity of the time-period, default 0.1
        
        Returns:
            s: A NumPy array with the simulated stock prices over the time-period deltaT
        """
        n_step = int(deltaT/dt)
        time_vector = np.linspace(0,deltaT,num=n_step)
        # Stock variation
        stock_var = (mu-(sigma**2/2))*time_vector
        # Forcefully set the initial value to zero for the stock price simulation
        self.x0=0
        # Weiner process (calls the `gen_normal` method)
        weiner_process = sigma*self.gen_normal(n_step)
        # Add two time series, take exponent, and multiply by the initial stock price
        s = s0*(np.exp(stock_var+weiner_process))
        
        return s
    
    
