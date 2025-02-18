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
    
    def __repr__(self):
        return (
            f"{self.symbol}: {self.balance}"
            f"\nPrice: {self.price}"
            )
    
    def __str__(self):
        return f"{self.symbol}: {self.balance}"


class LiquidityPool:
    def __init__(self, token_x: Token, token_y: Token, tick_spacing=100):
        """
        Native price is y/x price or else liquidity calculations won't work
        token x and y need to be selected such that when the lower range is hit, it is all token x and the upper range is all token y
        """
        # TODO: Add option to harvest fees instead of compounding them automatically
        self.token_x = token_x
        self.token_y = token_y
        self.liquidity = None
        self.lower_tick = None
        self.upper_tick = None
        self.init_x_bal = None
        self.init_x_price = None
        self.init_y_bal = None
        self.init_y_price = None
        self.seed = None
        self.apr = 0
        self.initial_setup = True
        self.duration = 0
        self.fees_accrued = 0
        self.total_fees = 0
        self.tick_spacing = tick_spacing
        self.dust = 0
        self.gm_rebalance = False
        self.gm_start_tick = None
        self.compound = False
        self.tick_offset_tracker = []  # track the various ranges used for gm rebalancing
        self.price_range_tracker = []
        self.rebalance_target_tick = None
    
    @property
    def value(self):
        return self.token_x.value + self.token_y.value
    
    @property
    def value_plus_fees(self):
        return self.value + self.fees_accrued + self.total_fees
    
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
            seed = self.seed - self.dust  # Just assume dust wasn't there in the beginning to give a fair comparison
            if int(self.token_x.price) == 1:
                return (seed / self.init_y_price) * self.token_y.price
            elif int(self.token_y.price) == 1:
                return (seed / self.init_x_price) * self.token_x.price
            else:
                # TODO: the dust calculations will not be correct for non stable-based pairs.  Need to think about this.
                return self.init_x_bal * self.token_x.price + self.init_y_bal * self.token_y.price
            
    @property
    def impermanent_loss(self):
        """
        Negative value means that holding the LP leads to "impermanent gains"
        """
        return self.hold_value - self.value_plus_fees

    @property
    def impermanent_gain(self):
        return -1 * self.impermanent_loss
        
    @property
    def ratio(self):
        ratio = (self.current_tick - self.lower_tick) / (self.upper_tick - self.lower_tick)
        if ratio > 0:
            return ratio
        else: 
            return None
    
    @property
    def current_tick(self):
        """
        This is the current price tick
        """
        return um.price_to_tick(self.native_price)
    
    @property
    def current_liq_tick(self):
        """
        This is the tick that ranges will be based on.  It's hte starting point of the bin that the current price is in
        """
        return self.current_tick // self.tick_spacing * self.tick_spacing
    
    @property
    def current_liq_tick_price(self):
        """
        This is the price at the current liquidity tick
        """
        return um.sqrtp_to_price(um.tick_to_sqrtp(self.current_tick // self.tick_spacing * self.tick_spacing))
    
    @property
    def range(self):
        return [um.sqrtp_to_price(um.tick_to_sqrtp(self.lower_tick)), um.sqrtp_to_price(um.tick_to_sqrtp(self.upper_tick))]
    
    @property
    def in_range(self):
        if self.range[0] <= self.native_price <= self.range[1]:
            return True
        else:
            return False
        
    @property
    def gm_return_range(self):
        """
        Keep track of the range for when we should narrow the range again
        This needs to be not just when it's back within the range, but closer to the GM
        Perhaps it should be half of the original range?
        """
        if len(self.price_range_tracker) >= 2:
            # Use half the original range
            r = self.price_range_tracker[0][1] - self.price_range_tracker[0][0]
            mid = np.mean(self.price_range_tracker[0])
            return [mid - r/2/2, mid + r/2/2]
        else:
            return None
        
    def setup_new_position(self, seed, ticks_lower, ticks_higher):
        # Ticks are fixed based on the liquidity tick, not the current tick (could be in between)
        self.upper_tick = self.current_liq_tick + ticks_higher * self.tick_spacing
        self.lower_tick = self.current_liq_tick - ticks_lower * self.tick_spacing
        self.gm_start_tick = self.current_liq_tick
        self.tick_offset_tracker.append([ticks_lower, ticks_higher])
        self.price_range_tracker.append(self.range)
        self.add_liquidity(seed)
        # Make note of the initial token values to look at impermanent loss
        if self.initial_setup:
            self.seed = self.value
            self.init_x_price = self.token_x.price
            self.init_y_price = self.token_y.price
            self.init_x_bal = self.token_x.balance
            self.init_y_bal = self.token_y.balance
            self.initial_setup = False

    def add_liquidity(self, seed):
        # As price moves to the upper range (and therefore the "ratio"), the amount of x is decreasing and the amount of y is increasing
        token_x_bal = seed * (1 - self.ratio) / self.token_x.price
        token_y_bal = seed * self.ratio / self.token_y.price
        # Calculate liquidity
        liq_x = um.liquidity_x(token_x_bal, self.native_price, self.range[1])
        liq_y = um.liquidity_y(token_y_bal, self.native_price, self.range[0])
        self.liquidity = min(liq_x, liq_y)
        self.update_token_balances(0)  # need to align with ticks, not always even
        if not self.initial_setup:
            self.dust += seed - self.value  # add to dust
        
    def update_token_balances(self, duration, fee_per_ut_per_tick=0):
        self.duration += duration  # add duration days
        # Check to make sure not out of range.  If out, set the price to the boundary edge to calculate the token amount
        if self.native_price < self.range[0]:
            price = self.range[0]
        elif self.native_price > self.range[1]:
            price = self.range[1]
        else:
            price = self.native_price
            # Accure fees if in range
            self.fees_accrued  += fee_per_ut_per_tick / sum(self.tick_offset_tracker[-1])
        self.token_x.balance = um.calc_amount_x(self.liquidity, price, self.range[1])
        self.token_y.balance = um.calc_amount_y(self.liquidity, price, self.range[0])
        self.calc_apr_for_duration()

    def rebalance(self, ticks_lower, ticks_higher):
        if self.gm_rebalance:
            current_liq_tick = self.gm_start_tick
        else:
            current_liq_tick = self.current_liq_tick
        self.upper_tick = current_liq_tick + ticks_higher * self.tick_spacing
        self.lower_tick = current_liq_tick - ticks_lower * self.tick_spacing
        self.tick_offset_tracker.append([ticks_lower, ticks_higher])
        self.price_range_tracker.append(self.range)
        if self.compound:
            seed = self.value_plus_fees * 0.9999  # VFAT charges 0.01% fees on the balance
            self.fees_accrued = 0  # fees get compounded into the new balance
        else:
            self.withdraw_fees_accrued()  # Harvest Fees
            seed = self.value * 0.9999  # VFAT charges 0.01% fees on the balance
        self.add_liquidity(seed)

    def calc_apr_for_duration(self):
        if self.duration > 0 and self.impermanent_loss > 0:
            self.apr = (self.impermanent_loss / self.value) * (365 / self.duration) * 100
        else:
            self.apr = 0

    def coumpound_fees_accured(self):
        seed = self.value_plus_fees
        self.total_fees += self.fees_accrued
        self.fees_accrued = 0 
        self.add_liquidity(seed)
        self.calc_apr_for_duration()

    def withdraw_fees_accrued(self):
        self.total_fees += self.fees_accrued
        self.fees_accrued = 0

    def __repr__(self):
        return (
            f"{self.token_x.symbol}: {self.token_x.balance:.6f} (${self.token_x.value:.2f})| {self.token_y.symbol}: {self.token_y.balance:.6f} (${self.token_y.value:.2f})\n" 
            f"LP Value: ${self.value:.2f}, Fees Collected: ${self.total_fees:.2f}\n"
            f"Current Price: {self.native_price:.6f} in {self.token_y.symbol}/{self.token_x.symbol}\n"
            f"Ticks (low, current, high) {self.lower_tick}, {self.current_tick}, {self.upper_tick}\n"
            f"Range: {self.range[0]:.6f} ~ {self.range[1]:.6f}\n"
            f"Hold Value: ${self.hold_value:.2f}\n"
            f"Impermanent Loss: ${self.impermanent_loss:.2f}\n"
            f"Fee APR required to offset IL: {self.apr:.1f}%\n"
            f"Fees Accrued: ${self.fees_accrued:.2f}\n"
            f"Total Fees: ${self.total_fees:.2f}\n"
            f"Dust returned: ${self.dust:.2f}\n"
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
        # print(f"Try #{increment}")
        try:
            time.sleep(1)
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
    
def apy_to_apr(apy, n):
    """
    n = compounding periods
    """
    apr = ((apy/100 + 1) ** (1/n) - 1) * n
    return apr * 100
    
