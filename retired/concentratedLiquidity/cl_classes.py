import pickle
import time
import numpy as np
from api import bsc_url, cmc_key, base_url
from web3 import Web3
import json
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import date
from sklearn.preprocessing import PolynomialFeatures
import json
import unimath


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


def get_uniswap_v3_pool_volume(pool_address, days=7):
    """
    Unfortunately, the uniswap sub-graph only provides information for main-net.
    There is not a graph for base or any of the other L2 networks that I can find.
    """
    # Construct the GraphQL query
    query = """
    {
      pool(id: "%s") {
        id
        token0 {
          symbol
        }
        token1 {
          symbol
        }
        liquidity
        volumeUSD
        feeTier
        poolDayData(first: %d, orderBy: date, orderDirection: desc) {
          date
          volumeUSD
        }
      }
    }
    """ % (pool_address.lower(), days)

    # Send the GraphQL query to the Uniswap V3 subgraph
    response = requests.post('https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3-arbitrum',
                             json={'query': query})

    # Parse the response
    if response.status_code == 200:
        data = response.json()
        return data['in_data']['pool']
    else:
        raise Exception(f"Query failed with status code {response.status_code}")


class SimpleToken:
    """
    Simple token to just track a few parameters like name, price, balance, etc.
    """

    def __init__(self, name, balance, price_usd, token_addr, x_or_y, base_or_quote):
        self.name = name
        self.balance = balance
        self.price_usd = price_usd
        self.token_addr = token_addr
        self.x_or_y = x_or_y
        self.base_or_quote = base_or_quote


class V3DexLP:
    """
    Get LP information from Dex Screener
    The quote token is the "backing" token, typically ETH or a stable coin
    """

    def __init__(self, pair_addr: str, chain: str, coingecko_id: str, start_date: pd.Timestamp,
                 end_date: pd.Timestamp, historical_base='usd'):
        self.pair_addr = pair_addr
        self.chain = chain
        self.fee_tier = None
        # Query DexScreener
        url = f"https://api.dexscreener.io/latest/dex/pairs/{self.chain}/{self.pair_addr}"
        print(f"Getting Info for {self.chain} pair {self.pair_addr}")
        self.response = requests.get(url)
        if self.response.status_code == 200:
            self.pair_info = self.response.json()
        else:
            raise Exception(f"Failed to fetch in_data. Status code: {self.response.status_code}")
        # Create Tokens
        self.base_token = SimpleToken(self.pair_info['pairs'][0]['baseToken']['symbol'],
                                      self.pair_info['pairs'][0]['liquidity']['base'],
                                      float(self.pair_info['pairs'][0]['priceUsd']),
                                      self.pair_info['pairs'][0]['baseToken']['address'], None,
                                      'base')
        self.quote_token = SimpleToken(self.pair_info['pairs'][0]['quoteToken']['symbol'],
                                       self.pair_info['pairs'][0]['liquidity']['quote'],
                                       float(self.pair_info['pairs'][0]['priceUsd']) /
                                       float(self.pair_info['pairs'][0]['priceNative']),
                                       self.pair_info['pairs'][0]['quoteToken']['address'], None,
                                       'quote')
        self.liquidity_usd = self.pair_info['pairs'][0]['liquidity']['usd']
        self.day_volume = self.pair_info['pairs'][0]['volume']['h24']
        [self.hist_prices, self.hist_volumes] = get_historical_prices(coingecko_id, start_date,
                                                                      end_date, historical_base)
        self.tick_spacing = None
        self.pool_liquidity = self.get_liquidity_ticks()
        self.filename = f"in_data/{self.base_token.name}_{self.fee_tier}.pkl"
        self.save_to_pickle()

    def get_liquidity_ticks(self):
        web3 = Web3(Web3.HTTPProvider(base_url))
        with open('univ3contract_abi.json', 'r') as abi_file:
            contract_abi = json.load(abi_file)
        pool_contract = web3.eth.contract(address=self.pair_addr, abi=contract_abi)
        # Fetch the current tick from slot0
        slot0 = pool_contract.functions.slot0().call()
        current_price_tick = slot0[1]
        lp_price = unimath.sqrtp_to_price(unimath.tick_to_sqrtp(current_price_tick))
        self.fee_tier = pool_contract.functions.fee().call() / 1E6
        # Get tick spacing
        try:
            self.tick_spacing = pool_contract.functions.tickSpacing().call()
        except Exception as e:
            print(f"Error fetching tick spacing: {e}")
            return 0
        # Get which tokens are "x" and "y" in the LP
        token_x_addr = pool_contract.functions.token0().call()
        token_y_addr = pool_contract.functions.token1().call()
        if token_x_addr == self.base_token.token_addr and token_y_addr == self.quote_token.token_addr:
            self.base_token.x_or_y = 'x'
            self.quote_token.x_or_y = 'y'
        elif token_x_addr == self.quote_token.token_addr and token_y_addr == self.base_token.token_addr:
            self.quote_token.x_or_y = 'x'
            self.base_token.x_or_y = 'y'
        else:
            raise Exception("Dex token addresses do not match Uniswap LP token addresses")
        current_liquidity_tick = current_price_tick // self.tick_spacing * self.tick_spacing
        # Get 5 liquidity ticks above and 4 below the current one.  Go in order from middle out
        tick_refs = range(-9, 11, 1)
        ticks_to_get = []
        for ref in tick_refs:
            ticks_to_get.append(ref * self.tick_spacing + current_liquidity_tick)
        liquidity = {}
        i = 0
        while i < len(ticks_to_get):
            try:
                result = pool_contract.functions.ticks(ticks_to_get[i]).call()
                if result[0] != 0:
                    liquidity[ticks_to_get[i]] = result[0]
                    print(f"Liquidity at tick {ticks_to_get[i]} is {result[0]}")
                else:
                    print(f"Liquidity at tick {ticks_to_get[i]} is {result[0]}, therefore skipping")
                time.sleep(0.25)
                i += 1
            except Exception as e:
                print(f"Error fetching info for tick {ticks_to_get[i]}: {e}")
                print("wait 5 seconds and retry")
                time.sleep(5)
        return pd.Series(liquidity).astype('float64')

    def save_to_pickle(self):
        print(f"Saving to {self.filename}")
        with open(self.filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_from_pickle(filename):
        print(f"Loading info from {filename}")
        with open(filename, 'rb') as file:
            return pickle.load(file)


class LiquidityPosition:

    def __init__(self, base_lp: V3DexLP):
        # Set up LP liquidity ticks
        self.pool_liquidity = base_lp.pool_liquidity
        self.tick_spacing = base_lp.tick_spacing
        # Set up generic LP info
        self.dex_lp = base_lp
        self.fee_tier = base_lp.fee_tier
        self.gas_fee = 0.20  # Cost for a re-balance or compound
        # Coingecko historical in_data is for all LPs, for a more accurate fee calc, try to match the ratios
        self.pool_ratio = base_lp.day_volume / (self.dex_lp.hist_volumes.iloc[-1] *
                                                self.dex_lp.quote_token.price_usd)
        if self.dex_lp.base_token.x_or_y == 'y':
            # Coingecko price is either going to be in the quote token (like ETH or USD), therefore if true,
            # the price needs to be inverted to match the raw price in the Uniswap LP
            # Otherwords, the quote token needs to be 'y' for y/x pricing or it needs to be inverted
            self.invert_price = True
        else:
            self.invert_price = False
        if base_lp.base_token.x_or_y == 'x':
            token1 = SimpleToken(base_lp.base_token.name, 0, 0,
                                 base_lp.base_token.token_addr, 'x', 'base')
            token2 = SimpleToken(base_lp.quote_token.name, 0, base_lp.quote_token.price_usd,
                                 base_lp.quote_token.token_addr, 'y', 'quote')
        else:
            token1 = SimpleToken(base_lp.quote_token.name, 0, base_lp.quote_token.price_usd,
                                 base_lp.quote_token.token_addr, 'x', 'quote')
            token2 = SimpleToken(base_lp.base_token.name, 0, 0,
                                 base_lp.base_token.token_addr, 'y', 'base')
        # Creating the dict makes an easy way to reference tokens by either base/quote, or x/y
        self.tokens = {
            'x': token1 if token1.x_or_y == 'x' else token2,
            'y': token1 if token1.x_or_y == 'y' else token2,
            'base': token1 if token1.base_or_quote == 'base' else token2,
            'quote': token1 if token1.base_or_quote == 'quote' else token2
        }
        self.liquidity = None
        self.liquidity_per_tick = None
        self.price = None
        self.upper_range = None  # Upper range is P_b or when x = 0
        self.lower_range = None  # Lower range is P_a or when y = 0

    @property
    def position_usd(self):
        base_balance = self.tokens['base'].balance * 1E12 if self.dex_lp.base_token.name == 'USDC' else self.tokens[
            'base'].balance
        quote_balance = self.tokens['quote'].balance * 1E12 if self.dex_lp.quote_token.name == 'USDC' else self.tokens[
            'quote'].balance
        return base_balance * self.usd_price + quote_balance * self.dex_lp.quote_token.price_usd

    @property
    def low_tick(self):
        # Also align with liquidity
        return unimath.price_to_tick(self.lower_range) // self.tick_spacing * self.tick_spacing

    @property
    def high_tick(self):
        # Also align with liquidity ticks
        return unimath.price_to_tick(self.upper_range) // self.tick_spacing * self.tick_spacing

    @property
    def position_tick_range(self):
        return self.get_liquidity_tick_range(self.low_tick, self.high_tick)

    @property
    def usd_price(self):
        return self.price * self.dex_lp.quote_token.price_usd

    @property
    def lp_price(self):
        if self.invert_price:
            return 1 / self.price
        if self.dex_lp.quote_token.name == 'USDC' or self.dex_lp.base_token.name == 'USDC':
            return self.price * 1E-12  # USDC only uses 6 decimals vs 18 for other ethereum based tokens
        else:
            return self.price

    def convert_dexprice_to_lpprice(self, price):
        if self.invert_price:
            return 1 / price
        if self.dex_lp.quote_token.name == 'USDC' or self.dex_lp.base_token.name == 'USDC':
            return price * 1E-12  # USDC only uses 6 decimals vs 18 for other ethereum based tokens
        else:
            return price

    def get_price_ranges(self, low_pct, high_pct):
        # We need to convert from USD "prices" into LP "prices"
        # if the price is inverted relative to USD, we want the lower range to match the "high_pct"
        if self.invert_price:
            # The higher "price" in this case is actually a smaller number
            self.upper_range = self.lp_price * (1 + low_pct)
            self.lower_range = self.lp_price * (1 / (1 + high_pct))
        else:
            self.upper_range = self.lp_price * (1 + high_pct)
            self.lower_range = self.lp_price * (1 / (1 + low_pct))

    def calc_liquidity_from_seed(self, seed):
        # The base and quote token ratios, need to be mindful of the price inversion
        ratio = (self.lp_price - self.lower_range) / (self.upper_range - self.lower_range) if self.invert_price \
            else (self.upper_range - self.lp_price) / (self.upper_range - self.lower_range)
        # The closer the price to the lower range, the more of the "base" token is needed
        self.tokens['quote'].balance = seed * (1 - ratio) / self.dex_lp.quote_token.price_usd
        self.tokens['quote'].balance = self.tokens['quote'].balance * 1E-12 if self.dex_lp.quote_token.name == 'USDC' \
            else self.tokens['quote'].balance  # handle usdc
        self.tokens['base'].balance = seed * ratio / self.usd_price
        self.tokens['base'].balance = self.tokens['base'].balance * 1E-12 if self.dex_lp.base_token.name == 'USDC' \
            else self.tokens['base'].balance  # handle usdc
        # re-calc upper range based on amounts to make liquidity even
        self.upper_range = unimath.calc_upper_range_pb(self.tokens['x'].balance, self.tokens['y'].balance,
                                                       self.lower_range, self.lp_price)
        liq_x = unimath.liquidity_x(self.tokens['x'].balance, self.lp_price, self.upper_range)
        liq_y = unimath.liquidity_y(self.tokens['y'].balance, self.lp_price, self.lower_range)
        liquidity = min(liq_y, liq_x)
        liquidity_per_tick = liquidity / len(self.position_tick_range)
        return liquidity, liquidity_per_tick

    def re_calculate_balances(self):
        # When calculating the balances, if the price goes outside the range, the calculation fails.
        # When outside the range, one of the balances should be 0
        adjusted_price = self.adjust_to_bounds(self.lp_price)
        self.tokens['x'].balance = unimath.calc_amount_x(self.liquidity, adjusted_price, self.upper_range)
        self.tokens['y'].balance = unimath.calc_amount_y(self.liquidity, adjusted_price, self.lower_range)

    def adjust_to_bounds(self, price):
        lower_bound = min(self.lower_range, self.upper_range)
        upper_bound = max(self.lower_range, self.upper_range)

        if price < lower_bound:
            return lower_bound
        elif price > upper_bound:
            return upper_bound
        else:
            return price

    def check_in_range(self, price):
        lower_bound = min(self.lower_range, self.upper_range)
        upper_bound = max(self.lower_range, self.upper_range)
        check = True if lower_bound <= price <= upper_bound else False
        return check

    def tick_to_liq_tick(self, tick):
        return tick // self.tick_spacing * self.tick_spacing

    def get_liquidity_tick_range(self, current, previous):
        if previous > current:
            current, previous = previous, current
        return range(self.tick_to_liq_tick(previous), self.tick_to_liq_tick(current) + self.tick_spacing,
                     self.tick_spacing)

    def simulate_range(self, usd_investment: float, low_pct: float, high_pct: float, rebalance=False,
                       autocompound=False):
        self.price = self.dex_lp.hist_prices.iloc[0]
        self.get_price_ranges(low_pct, high_pct)
        self.liquidity, self.liquidity_per_tick = self.calc_liquidity_from_seed(usd_investment)
        self.re_calculate_balances()  # Always need to re-calc balances after new liquidity position
        usd_investment = self.position_usd  # Update the starting position after getting liquidity calcs completed
        # Set up variables for the loop
        result = {}
        fees = []
        position_usd = []
        fees_accrued = 0
        fees_compounded = 0
        rebalances = 0
        for i in range(self.dex_lp.hist_prices.size - 1):
            i += 1  # Start after the first passed hour
            self.price = self.dex_lp.hist_prices.iloc[i]  # auto assigns usd and lp prices
            self.re_calculate_balances()
            position_usd.append(self.position_usd)
            # Handle Auto-Rebalance (automatically compounds fees)
            if not self.check_in_range(self.lp_price) and rebalance:
                # Get new price range around current spot and update liquidity ranges
                # Price is already corrected or swapped at this point.
                self.get_price_ranges(low_pct, high_pct)
                seed = (self.position_usd + fees_accrued - self.gas_fee)
                fees_compounded += fees_accrued
                fees_accrued = 0
                self.liquidity, self.liquidity_per_tick = self.calc_liquidity_from_seed(seed)
                self.re_calculate_balances()  # Always need to re-calc balances after new liquidity position
                fees_accrued += seed - self.position_usd  # Recover rounding errors from new position created
                rebalances += 1
            if not self.check_in_range(self.lp_price):
                continue  # Out of range.  Don't compound or calculate fees
            # Handle Auto-compound and set up new position
            if fees_accrued > usd_investment * 0.025 and autocompound:
                # Fees need to be added to the liquidity of our position
                seed = self.position_usd + fees_accrued - self.gas_fee
                fees_compounded += fees_accrued
                fees_accrued = 0
                self.liquidity, self.liquidity_per_tick = self.calc_liquidity_from_seed(seed)
                self.re_calculate_balances()  # Always need to re-calc balances after new liquidity position
            # Tick is in range, need to find liquidity ratios.  Use an average of the liquidity ticks for the whole pool
            previous_price = self.convert_dexprice_to_lpprice(self.dex_lp.hist_prices.iloc[i - 1])
            tick = unimath.price_to_tick(self.lp_price)
            previous_tick = unimath.price_to_tick(previous_price)
            liq_tick_range = self.get_liquidity_tick_range(previous_tick, tick)
            liq_ticks = len(liq_tick_range)
            total_liquidity = self.pool_liquidity.mean() * liq_ticks
            my_liquidity = self.liquidity_per_tick * liq_ticks
            ratio = my_liquidity / total_liquidity
            # Coingecko volume is a running 24-hour volume
            volume = self.dex_lp.hist_volumes.iloc[i] * self.pool_ratio / 24 * self.dex_lp.quote_token.price_usd
            fee_collected = volume * self.fee_tier * ratio
            fees.append(fee_collected)
            fees_accrued += fee_collected
        fees = pd.Series(fees)
        position_usd = pd.Series(position_usd)
        result['accrued_fees'] = fees_accrued
        result['compounded_fees'] = fees_compounded
        result['final_position_usd'] = position_usd.iloc[-1]
        result['reblances'] = rebalances
        result['total profit'] = result['final_position_usd'] + fees_accrued - usd_investment
        result['fee apr (%)'] = (fees_accrued + fees_compounded) / usd_investment * 100 * 365 / (
                self.dex_lp.hist_prices.size / 24)
        result['total apr (%)'] = result['total profit'] / usd_investment * 100 * 365 / (
                self.dex_lp.hist_prices.size / 24)
        # print(result)
        return result


def get_model_prediction(model: LinearRegression, delta_days):
    # first need to build input polynomial
    j = [0]
    for i in range(model.rank_):
        j.append(delta_days ** (i + 1))
    model_in = np.array(j).reshape(1, -1)  # Need to reshape into model's desired array format
    return model.predict(model_in)[0]


class FuturesModel:
    """
    Model of futures using ML (scikit-learn) to predict future prices based on downloaded dune in_data
    This model is at the aggregate level, not individual wallets
    """

    def __init__(self, deposit_model: LinearRegression, withdraw_model: LinearRegression,
                 compound_model: LinearRegression, model_start_date: str):
        self.deposit_model = deposit_model
        self.withdrawal_model = withdraw_model
        self.compound_model = compound_model
        self.start_date = pd.to_datetime(model_start_date)
        self.today = pd.to_datetime(pd.Timestamp.today().date())

    @property
    def delta_days(self):
        return (self.today - self.start_date).days

    @property
    def deposits(self):
        return get_model_prediction(self.deposit_model, self.delta_days)

    @property
    def deposit_delta(self):
        yesterday = get_model_prediction(self.deposit_model, self.delta_days - 1)
        return self.deposits - yesterday

    @property
    def withdrawals(self):
        return get_model_prediction(self.withdrawal_model, self.delta_days)

    @property
    def withdrawal_delta(self):
        yesterday = get_model_prediction(self.withdrawal_model, self.delta_days - 1)
        return self.withdrawals - yesterday

    @property
    def compounds(self):
        return get_model_prediction(self.compound_model, self.delta_days)

    @property
    def tvl(self):
        return self.deposits + self.compounds - self.withdrawals

    @property
    def claims(self):
        return self.compounds + self.withdrawals
