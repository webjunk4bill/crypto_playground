# Module imports
import numpy as np
from pandas import DataFrame

import core.helper_classes as hc
from core import farm
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td
import matplotlib.pyplot as plt
import pycoingecko
from aenum import Enum


def predict():
    # Keep the historical fetching below 90 days to get hourly based data
    api_id = 'bitcoin'
    end = pd.Timestamp(dt.today())
    start = end - pd.DateOffset(days=80)  # Get previous years worth of data
    btc_data = pd.DataFrame(hc.get_historical_prices(api_id, start, end)).T

    # api_id = 'solana'
    # end = pd.Timestamp(dt.today())
    # start = end - pd.DateOffset(days=80)  # Get previous years worth of data
    # sol_data = pd.DataFrame(hc.get_historical_prices(api_id, start, end)).T

    # Use Weiner process to get future price predictions
    lookback_days = 7  # Decide how much historical data to use for prediciton input
    lookback_hours = lookback_days * 24
    data = btc_data.copy().tail(lookback_hours)
    data['gain'] = data['price'].pct_change()
    mean_gain = data['gain'].mean()
    var_multiplier = 2  # use this to get a wider range of outcomes
    std_gain = data['gain'].std() * var_multiplier



    latest_btc_price = btc_data['price'].values[-1]
    btc = hc.Token("BTC", latest_btc_price)  # using latest price for  to project out
    iterations = 2500  # Can adjust this for more variance if desired
    predict_days = 14
    predict_hours = predict_days * 24
    b = hc.Brownian()
    predict = []
    for i in range(iterations):
        x = b.stock_price(btc.price, mean_gain, std_gain, predict_hours, 1)
        predict.append(x)
    df = pd.DataFrame(predict).T

    # Look at price prediction over time
    mean_price = df.mean(axis=1)
    min_price = df.min(axis=1)
    max_price = df.max(axis=1)
    std_price = df.std(axis=1)

    plt.plot(df.index, mean_price, label='Mean')
    plt.plot(df.index, min_price, label='Min')
    plt.plot(df.index, max_price, label='Max')
    plt.title('Bitcoin Price Prediction - Bounds')
    plt.xlabel('Time (hours)')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()

    df.plot(legend=False, ylabel='Price(USD)', xlabel='Hours',title='Bitcoin Price Prediction Simulations')

    return df, latest_btc_price

#prediction output, enable if you want to run the original method
# df, latest_btc_price = predict()

#real data

#todo: write a cleaner version, maybe just aggregate data and organize it.
latest_btc_price = 100000 # i.e. the price you start the position with.
data_file_a = "data/2025-02-10_BTC-USD_28d_1m.csv"
df1 = pd.read_csv(data_file_a)
data_file_b = "data/2025-02-10_BTC-USD_21d_1m.csv"
df2 = pd.read_csv(data_file_b)
data_file_c = "data/2025-02-10_BTC-USD_14d_1m.csv"
df3 = pd.read_csv(data_file_b)
data_file_d = "data/2025-02-10_BTC-USD_7d_1m.csv"
df4 = pd.read_csv(data_file_b)

df = pd.concat([df1, df2, df3, df4], ignore_index=True)
df.drop_duplicates(subset=["Date"], inplace=True, keep='first')
df.loc[:, "Price"].plot()
plt.title('Bitcoin Price Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Price (USD)')
plt.show()
latest_btc_price = df.loc[:,  "Price"].iloc[0]
print(latest_btc_price)



# cbBTC/USDC Farm stats on Aerodrome
# btc = farm.Farm(token_a='btc', token_b='USDC', weekly_reward=400652, tvl_reward=1.8E6)
# btc.set_seed(10000)
#todo:Zee's farm class, refactor.

# Hourly calc
# tick_spacing = 100
# weekly_rewards = 394411
# tvl_rewarded = 7E6  # This can change quite a bit and determines how "concentrated" the pool is
# apr_per_tick = weekly_rewards / tvl_rewarded * 52 * 100
# seed = 7500
# fee_per_ut_per_tick = apr_per_tick / 100 / 365 / 24 * seed
# print(f"Fee per hour, per liquidity tick: ${fee_per_hour_per_tick:.2f}")


# Minute calc
tick_spacing = 100
weekly_rewards = 440000
tvl_rewarded = 2.1E6  # This can change quite a bit and determines how "concentrated" the pool is
apr_per_tick = weekly_rewards / tvl_rewarded * 52 * 100
seed = 10000
fee_per_ut_per_tick = apr_per_tick / 100 / 365 / 24 / 60 * seed
print(f"Fee per UT, per liquidity tick: ${fee_per_ut_per_tick:.2f}")

class RangeMode(Enum):
    EVEN = 'EVEN'
    FIXL = 'FIXL'
    FIXH = 'FIXH'
    LTH = 'LTH'
    HTL = 'HTL'


# Decide on ranges to use.
export_data = []
min_tolerance = 2
max_tolerance = 15

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
        fee_per_ut = fee_per_ut_per_tick / (high_tick + low_tick + 1)
        print(f"Fee per UT: ${fee_per_ut:.2f}")

        il = None
        gains = None
        rebalance = True

        btc = hc.Token("BTC", latest_btc_price)
        usdc = hc.Token("USDC", 1)
        btc_usdc = hc.LiquidityPool(btc, usdc)
        btc_usdc.gm_rebalance = False
        btc_usdc.setup_new_position(seed, low_tick, high_tick)
        rebal_ctr = 0
        no_rebal_ctr = 0
        for price in df["Price"]:
            btc.price = price
            btc_usdc.update_token_balances(1/24/60)
            if btc_usdc.in_range:
                btc_usdc.fees_accrued += fee_per_ut * 0.991  # VFAT charges 0.9% fee on AERO rewards
                no_rebal_ctr += 1 
            elif rebalance:
                btc_usdc.rebalance(low_tick, high_tick)
                rebal_ctr += 1
            else:
                pass
        il = btc_usdc.impermanent_loss
        gains = btc_usdc.impermanent_gain
        loss = pd.Series(il)
        print(f'Gain from Simulation of Range +{high_pct}/-{low_pct} is {gains}')
        print(f"In range: {no_rebal_ctr}, Rebalances: {rebal_ctr}")






        # todo: Bill's sim. Need to align dataframes to consolidate
        # for label, sim in df.items():  # Each column is a simulation
        #     # Set up LP
        #     btc = hc.Token("BTC", latest_btc_price)
        #     usdc = hc.Token("USDC", 1)
        #     btc_usdc = hc.LiquidityPool(btc, usdc)
        #     btc_usdc.gm_rebalance = False
        #     btc_usdc.setup_new_position(seed, low_tick, high_tick)
        #     for price in sim.iloc[1:].values:  # start after the first hour
        #         btc.price = price
        #         btc_usdc.update_token_balances(1/24)
        #         if btc_usdc.in_range:
        #             btc_usdc.fees_accrued += fee_per_ut * 0.982  # VFAT charges 1.8% fee on AERO rewards
        #         elif rebalance:
        #             btc_usdc.rebalance(low_tick, high_tick)
        #         else:
        #             pass
        #     il.append(btc_usdc.impermanent_loss)
        #     gains.append(btc_usdc.impermanent_gain)
        # loss = pd.Series(il)
        # loss.hist(bins=50)
        # print(f"{range_mode} Report")
        # print(f'Minimum Gain from Simulation of Range +{high_pct}/-{low_pct} is {min(gains)}')
        # print(f'Mean Gain from Simulation of Range +{high_pct}/-{low_pct} is {np.mean(gains)}')
        # print(f'Maximum Gain from Simulation of Range +{high_pct}/-{low_pct} is {max(gains)}')