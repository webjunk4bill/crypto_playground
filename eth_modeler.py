# Module imports
import core.helper_classes as hc
from core import farm
from aenum import Enum
import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt


api_id = 'ethereum'
end = pd.Timestamp(dt.today())
start = end - pd.DateOffset(days=80)  # Get previous years worth of data
eth_data = pd.DataFrame(hc.get_historical_prices(api_id, start, end)).T

# Ethereum
# Use Weiner process to get future price predictions
lookback_days = 30  # Decide how much historical data to use for prediciton input
lookback_hours = lookback_days * 24
data = eth_data.copy().tail(lookback_hours)
data['gain'] = data['price'].pct_change()
mean_gain = data['gain'].mean()
var_multiplier = 1.2  # use this to get a wider range of outcomes
std_gain = data['gain'].std() * var_multiplier

latest_eth_price = eth_data['price'].values[-1]
iterations = 2500  # Can adjust this for more variance if desired
predict_days = 14
predict_hours = predict_days * 24
b = hc.Brownian()
predict = []
for i in range(iterations):
    x = b.stock_price(latest_eth_price, mean_gain, std_gain, predict_hours, 1)
    predict.append(x)
df = pd.DataFrame(predict).T

# Look at price prediction over time
mean_price = df.mean(axis=1)
min_price = df.min(axis=1)
max_price = df.max(axis=1)
std_price = df.std(axis=1)

plt.clf()
plt.plot(df.index, mean_price, label='Mean')
plt.plot(df.index, min_price, label='Min')
plt.plot(df.index, max_price, label='Max')
plt.title('Ethereum Price Prediction - Bounds')
plt.xlabel('Time (hours)')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# WETH/USDC Farm stats on Aerodrome
tick_spacing = 100
weekly_rewards = 1308872
tvl_rewarded = 6E6  # This can change quite a bit and determines how "concentrated" the pool is
apr_per_tick = weekly_rewards / tvl_rewarded * 52 * 100
seed = 7000
fee_per_hour_per_tick = apr_per_tick / 100 / 365 / 24 * seed
print(f"Fee per hour, per liquidity tick: ${fee_per_hour_per_tick:.2f}")

class RangeMode(Enum):
    EVEN = 'EVEN'
    FIXL = 'FIXL'
    FIXH = 'FIXH'
    LTH = 'LTH'
    HTL = 'HTL'

# Decide on ranges to use.
data = []
range_mode = RangeMode.FIXH
min_tolerance = 2
max_tolerance = 7
il_report = {"Mininmum IL": None,
             "Minimum IL Range": None,
             "Maximum IL": None,
             "Maximum IL Range": None}

# Decide on ranges to use.
for range_mode in [RangeMode.EVEN, RangeMode.LTH, RangeMode.FIXL, RangeMode.FIXH]:

    for r in range(min_tolerance, max_tolerance + 1):

        if range_mode == RangeMode.EVEN:
            high_pct = r
            low_pct = r
        elif range_mode == RangeMode.LTH:
            high_pct = r
            low_pct = high_pct - 2 if high_pct - 2 >= 2 else 2
        elif range_mode == RangeMode.FIXL:
            high_pct = r
            low_pct = 2
        elif range_mode == RangeMode.FIXH:
            high_pct = 2
            low_pct = r

        # Convert to ticks based on spacing
        high_tick = int(high_pct * tick_spacing / 100)
        low_tick = int(low_pct * tick_spacing / 100)
        fee_per_hour = fee_per_hour_per_tick / (high_tick + low_tick + 1)
        print(f"Fee per hour: ${fee_per_hour:.2f}")


        il = []
        gains = []
        val = []
        rebalance = True
        for label, sim in df.items():  # Each column is a simulation
            # Set up LP
            eth = hc.Token("ETH", latest_eth_price)
            usdc = hc.Token("USDC", 1)
            eth_usdc = hc.LiquidityPool(eth, usdc)
            eth_usdc.gm_rebalance = False
            eth_usdc.setup_new_position(seed, low_tick, high_tick)
            for price in sim.iloc[1:].values:  # start after the first hour
                eth.price = price
                eth_usdc.update_token_balances(1/24)
                if eth_usdc.in_range:
                    eth_usdc.fees_accrued += fee_per_hour * 0.982  # VFAT charges 1.8% fee on AERO rewards
                elif rebalance:
                    eth_usdc.rebalance(low_tick, high_tick)
                else:
                    pass
            il.append(eth_usdc.impermanent_loss)
            gains.append(eth_usdc.impermanent_gain)
            val.append(seed - eth_usdc.value)
        loss = pd.Series(il)
        loss.hist(bins=50)
        final_value = pd.Series(val)
        final_value.hist(bins=50)
        print(f"{range_mode} Report")
        print(f'Minimum Gain from Simulation of Range +{high_pct}/-{low_pct} is {min(gains)}')
        print(f'Mean Gain from Simulation of Range +{high_pct}/-{low_pct} is {np.mean(gains)}')
        print(f'Maximum Gain from Simulation of Range +{high_pct}/-{low_pct} is {max(gains)}')
        data.append({
            "Seed": seed,
            "Mode": range_mode,
            "Low Range": low_pct,
            "High Range": high_pct,
            "Minimum Gain": np.min(gains),
            "Mean Gain": np.mean(gains),
            "Max Gain": np.max(gains),
        })

pd.DataFrame(data).to_csv("mode_report.csv")