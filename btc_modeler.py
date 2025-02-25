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

# Constants
TICK_SPACING = 100
WEEKLY_REWARDS = 440000
TVL_REWARDED = 2.1E6  # This can change quite a bit and determines how "concentrated" the pool is
SEED = 10000
MIN_TOLERANCE = 2
MAX_TOLERANCE = 3
TOKEN0 = 'BTC'
TOKEN1 = 'USDC'

class RangeMode(Enum):
    EVEN = 'EVEN'
    FIXL = 'FIXL'
    FIXH = 'FIXH'
    LTH = 'LTH'
    HTL = 'HTL'

def load_data():
    data_files = [
        "data/2025-02-10_BTC-USD_28d_1m.csv",
        "data/2025-02-10_BTC-USD_21d_1m.csv",
        "data/2025-02-10_BTC-USD_14d_1m.csv",
        "data/2025-02-10_BTC-USD_7d_1m.csv"
    ]
    dfs = [pd.read_csv(file, parse_dates=['Date']) for file in data_files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.set_index("Date")
    df.drop_duplicates(keep='first', inplace=True)
    return df

def plot_price(df):
    # Resample the data to a daily frequency to make the plot cleaner
    # df_resampled = df.resample('4h').mean()

    # Plot the resampled data
    # df_resampled.loc[:, "Price"].plot()
    df.loc[:, "Price"].plot()
    plt.title('Bitcoin Price Over Time')
    plt.xlabel('Time (days)')
    plt.ylabel('Price (USD)')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

def plot_results(slices, result_df):
    # Create the first plot with slices 1 and 2
    fig1, axs1 = plt.subplots(2, 2, figsize=(15, 10))
    axs1[0, 0].plot(slices[0].index, slices[0]['Price'], label='Slice 1')
    axs1[0, 0].set_title('Slice 1 Price')
    axs1[0, 0].set_xlabel('Date')
    axs1[0, 0].set_ylabel('Price')
    axs1[0, 0].legend()

    axs1[0, 1].plot(slices[1].index, slices[1]['Price'], label='Slice 2')
    axs1[0, 1].set_title('Slice 2 Price')
    axs1[0, 1].set_xlabel('Date')
    axs1[0, 1].set_ylabel('Price')
    axs1[0, 1].legend()

    # Filter results for slices 1 and 2
    slice1_results = result_df[result_df['date_range'].str.contains(slices[0].index[0].strftime('%Y-%m-%d'))]
    slice2_results = result_df[result_df['date_range'].str.contains(slices[1].index[0].strftime('%Y-%m-%d'))]

    # Plot gains for slice 1
    slice1_false = slice1_results[slice1_results['geometric_mean_rebalance'] == False]
    slice1_true = slice1_results[slice1_results['geometric_mean_rebalance'] == True]
    axs1[1, 0].plot(slice1_false['range'], slice1_false['gains'], label='Geometric Mean Rebalance: False')
    axs1[1, 0].plot(slice1_true['range'], slice1_true['gains'], label='Geometric Mean Rebalance: True')
    axs1[1, 0].set_title('Slice 1 Gains')
    axs1[1, 0].set_xlabel('Range')
    axs1[1, 0].set_ylabel('Gains')
    axs1[1, 0].legend()

    # Plot gains for slice 2
    slice2_false = slice2_results[slice2_results['geometric_mean_rebalance'] == False]
    slice2_true = slice2_results[slice2_results['geometric_mean_rebalance'] == True]
    axs1[1, 1].plot(slice2_false['range'], slice2_false['gains'], label='Geometric Mean Rebalance: False')
    axs1[1, 1].plot(slice2_true['range'], slice2_true['gains'], label='Geometric Mean Rebalance: True')
    axs1[1, 1].set_title('Slice 2 Gains')
    axs1[1, 1].set_xlabel('Range')
    axs1[1, 1].set_ylabel('Gains')
    axs1[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # Create the second plot with slices 3 and 4
    fig2, axs2 = plt.subplots(2, 2, figsize=(15, 10))
    axs2[0, 0].plot(slices[2].index, slices[2]['Price'], label='Slice 3')
    axs2[0, 0].set_title('Slice 3 Price')
    axs2[0, 0].set_xlabel('Date')
    axs2[0, 0].set_ylabel('Price')
    axs2[0, 0].legend()

    axs2[0, 1].plot(slices[3].index, slices[3]['Price'], label='Slice 4')
    axs2[0, 1].set_title('Slice 4 Price')
    axs2[0, 1].set_xlabel('Date')
    axs2[0, 1].set_ylabel('Price')
    axs2[0, 1].legend()

    # Filter results for slices 3 and 4
    slice3_results = result_df[result_df['date_range'].str.contains(slices[2].index[0].strftime('%Y-%m-%d'))]
    slice4_results = result_df[result_df['date_range'].str.contains(slices[3].index[0].strftime('%Y-%m-%d'))]

    # Plot gains for slice 3
    slice3_false = slice3_results[slice3_results['geometric_mean_rebalance'] == False]
    slice3_true = slice3_results[slice3_results['geometric_mean_rebalance'] == True]
    axs2[1, 0].plot(slice3_false['range'], slice3_false['gains'], label='Geometric Mean Rebalance: False')
    axs2[1, 0].plot(slice3_true['range'], slice3_true['gains'], label='Geometric Mean Rebalance: True')
    axs2[1, 0].set_title('Slice 3 Gains')
    axs2[1, 0].set_xlabel('Range')
    axs2[1, 0].set_ylabel('Gains')
    axs2[1, 0].legend()

    # Plot gains for slice 4
    slice4_false = slice4_results[slice4_results['geometric_mean_rebalance'] == False]
    slice4_true = slice4_results[slice4_results['geometric_mean_rebalance'] == True]
    axs2[1, 1].plot(slice4_false['range'], slice4_false['gains'], label='Geometric Mean Rebalance: False')
    axs2[1, 1].plot(slice4_true['range'], slice4_true['gains'], label='Geometric Mean Rebalance: True')
    axs2[1, 1].set_title('Slice 4 Gains')
    axs2[1, 1].set_xlabel('Range')
    axs2[1, 1].set_ylabel('Gains')
    axs2[1, 1].legend()

    plt.tight_layout()
    plt.show()

def calculate_fee_per_ut_per_tick():
    apr_per_tick = WEEKLY_REWARDS / TVL_REWARDED * 52 * 100
    fee_per_ut_per_tick = apr_per_tick / 100 / 365 / 24 / 60 * SEED
    print(f"Fee per UT, per liquidity tick: ${fee_per_ut_per_tick:.2f}\n")
    return fee_per_ut_per_tick

def simulate_range_mode(df, range_mode, fee_per_ut_per_tick, gm_rebalance):
    result = {}
    for r in range(MIN_TOLERANCE, MAX_TOLERANCE + 1):
        high_pct, low_pct = get_high_low_pct(range_mode, r)
        high_tick, low_tick = get_ticks(high_pct, low_pct)
        result.update(simulate_range(df, high_tick, low_tick, fee_per_ut_per_tick, gm_rebalance))
    return result

def get_high_low_pct(range_mode, r):
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
    return high_pct, low_pct

def get_ticks(high_pct, low_pct):
    high_tick = int(high_pct * TICK_SPACING / 100)
    low_tick = int(low_pct * TICK_SPACING / 100)
    return high_tick, low_tick

def simulate_range(df, high_tick, low_tick, fee_per_ut_per_tick, gm_rebalance):
    result = {}
    result[f"+{high_tick}/-{low_tick}"] = {}
    btc = hc.Token(TOKEN0, df.loc[:, "Price"].iloc[0])
    usdc = hc.Token(TOKEN1, 1)
    lp = hc.LiquidityPool(btc, usdc)
    lp.gm_rebalance = gm_rebalance
    lp.compound = False
    lp.setup_new_position(SEED, low_tick, high_tick)
    rebal_ctr = 0
    in_range_ctr = 0
    time_to_rebal_ctr = 0
    out_of_range_ctr = 0
    if gm_rebalance:
        time_to_rebalance = 0  # minutes out of range before a rebalance can occur
    else:
        time_to_rebalance = 0  # minutes out of range before a rebalance can occur

    for price in df["Price"]:
        if not len(lp.price_range_tracker) == len(lp.tick_offset_tracker):
            raise Exception("Tick Offset Tracker and Price Trackers have diverged")
        btc.price = price
        lp.update_token_balances(1/24/60, fee_per_ut_per_tick)
        if lp.in_range:
            in_range_ctr += 1
            time_to_rebal_ctr = 0
            if lp.gm_return_range:
                if lp.gm_return_range[0] <= price <= lp.gm_return_range[1]:
                    low = lp.tick_offset_tracker[-2][0]
                    high = lp.tick_offset_tracker[-2][1]
                    lp.tick_offset_tracker.pop()
                    lp.tick_offset_tracker.pop()
                    lp.price_range_tracker.pop()
                    lp.price_range_tracker.pop()
                    lp.rebalance(low, high)
                    rebal_ctr += 1
                    time_to_rebal_ctr = 0
        else:
            out_of_range_ctr += 1
            time_to_rebal_ctr += 1
            if time_to_rebal_ctr >= time_to_rebalance:
                if lp.gm_rebalance:
                    low1 = lp.tick_offset_tracker[0][0]
                    high1 = lp.tick_offset_tracker[0][1]
                    low2 = lp.tick_offset_tracker[-1][0]
                    high2 = lp.tick_offset_tracker[-1][1]
                    lp.rebalance(low1 + low2, high1 + high2)
                    rebal_ctr += 1
                    time_to_rebal_ctr = 0
                else:
                    lp.rebalance(low_tick, high_tick)
                    rebal_ctr += 1
                    time_to_rebal_ctr = 0

    lp.withdraw_fees_accrued()
    gains = lp.impermanent_gain
    print(lp.fetch_il_tracker())
    print(f'Gain from Simulation of Range +{high_tick}/{low_tick} is ${gains:.2f}')
    print(f"In range: {in_range_ctr}, Rebalances: {rebal_ctr}, Out of Range: {out_of_range_ctr}")
    print(f"Total Fees Collected: ${lp.total_fees:.2f} | Dust returned: ${lp.dust:.2f}")
    days_run = len(df["Price"]) / 60 / 24
    print(f"Average Fee APR: {lp.total_fees / lp.value * 100 * 365 / days_run:.1f}")
    print(f"Original Seed: ${SEED - lp.dust:.2f} | LP value plus fees: ${lp.value_plus_fees:.2f} | Hold Value ${lp.hold_value:.2f}\n")
    # result[f"+{high_tick}/-{low_tick}"][gm_rebalance] = gains
    return gains

def main():
    result_list = []
    price_df = load_data()
    plot_price(price_df)
    latest_btc_price = price_df.loc[:, "Price"].iloc[0]
    print(f"\nSimulation Starting Price: ${latest_btc_price:.2f}")
    fee_per_ut_per_tick = calculate_fee_per_ut_per_tick()

    # Split the DataFrame into 4 equal slices
    num_slices = 4
    slice_length = len(price_df) // num_slices
    slices = [price_df.iloc[i*slice_length:(i+1)*slice_length] for i in range(num_slices)]

    for range_mode in [RangeMode.EVEN]:  #, RangeMode.LTH, RangeMode.FIXL, RangeMode.FIXH]:
        for geo_mean_rebalance in [False, True]:
            for r in range(MIN_TOLERANCE, MAX_TOLERANCE + 1):
                high_pct, low_pct = get_high_low_pct(range_mode, r)
                high_tick, low_tick = get_ticks(high_pct, low_pct)
                slice_ctr = 0
                for slice_df in slices:
                    gains = simulate_range(slice_df, high_tick, low_tick, fee_per_ut_per_tick, geo_mean_rebalance)
                    result_list.append({
                        'date_slice': slice_ctr,
                        'date_range': f"{slice_df.index[0]} -> {slice_df.index[-1]}",
                        'geometric_mean_rebalance': geo_mean_rebalance,
                        'range': f"+{high_tick}/-{low_tick}",
                        'gains': gains
                        })
                    slice_ctr += 1
    print(f"Simulation Ending Price: ${price_df.loc[:, "Price"].iloc[-1]:.2f}")
    result_df = pd.DataFrame(result_list)
    print(result_df)
    result_df.to_csv("outputs/sim_result.csv")
    plot_results(slices, result_df)

if __name__ == "__main__":
    main()