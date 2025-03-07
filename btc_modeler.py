# Module imports
import core.helper_classes as hc
import pandas as pd
import matplotlib.pyplot as plt
from aenum import Enum
import glob
from concurrent.futures import ProcessPoolExecutor

# Constants
TICK_SPACING = 100
WEEKLY_REWARDS = 350000
TVL_REWARDED = 2.1E6  # This can change quite a bit and determines how "concentrated" the pool is
AVG_BINANCE_VOLUME = 21503974197 # Average weekly binance volume over the 7 day reward time period.  Can use this to scale rewards if desired
SEED = 10000
MIN_TOLERANCE = 2
MAX_TOLERANCE = 20
TOKEN0 = 'btc'
TOKEN1 = 'USDC'
MANUAL = True
MANUAL_RANGES = [
    [2, 20],
    [25, 25],
    [20, 20],
    [30, 30],
    [5, 5],
    [2, 2],
    [5, 25]
]

class RangeMode(Enum):
    EVEN = 'EVEN'
    FIXL = 'FIXL'
    FIXH = 'FIXH'
    LTH = 'LTH'
    HTL = 'HTL'
    MANUAL = 'MANUAL'

def plot_price(df):
    # Resample the data to a daily frequency to make the plot cleaner
    # df_resampled = df.resample('4h').mean()

    # Plot the resampled data
    # df_resampled.loc[:, "price"].plot()
    df.loc[:, "price"].plot()
    plt.title('Bitcoin Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()

def calculate_fee_per_ut_per_tick():
    apr_per_tick = WEEKLY_REWARDS / TVL_REWARDED * 52 * 100
    fee_per_ut_per_tick = apr_per_tick / 100 / 365 / 24 / 60 * SEED
    # print(f"Fee per UT, per liquidity tick: ${fee_per_ut_per_tick:.2f}\n")
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
    gain_v_time = []
    result[f"+{high_tick}/-{low_tick}"] = {}
    btc = hc.Token(TOKEN0, df.loc[:, "price"].iloc[0])
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
        time_to_rebalance = 30  # minutes out of range before a rebalance can occur

    print_ctr = 0 
    for price in df["price"]:
        print_ctr += 1
        if print_ctr % 43200 == 0:
            print(f"{int(print_ctr/43200)} months processed")
        if not len(lp.price_range_tracker) == len(lp.tick_offset_tracker):
            raise Exception("Tick Offset Tracker and Price Trackers have diverged")
        btc.price = price
        lp.update_token_balances(1/24/60, fee_per_ut_per_tick)
        gain_v_time.append(lp.clp_gain)
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
    gains = lp.clp_gain
    apr_required = lp.apr
    #print(lp.fetch_il_tracker())
    print(f'Gain from Simulation of Range +{high_tick}/{low_tick} is ${gains:.2f}')
    #print(f"In range: {in_range_ctr}, Rebalances: {rebal_ctr}, Out of Range: {out_of_range_ctr}")
    #print(f"Total Fees Collected: ${lp.total_fees:.2f} | Dust returned: ${lp.dust:.2f}")
    #days_run = len(df["price"]) / 60 / 24
    #print(f"Average Fee APR: {lp.total_fees / lp.value * 100 * 365 / days_run:.1f}")
    #print(f"Original Seed: ${SEED - lp.dust:.2f} | LP value plus fees: ${lp.value_plus_fees:.2f} | Hold Value ${lp.hold_value:.2f}\n")
    # result[f"+{high_tick}/-{low_tick}"][gm_rebalance] = gains
    return gains, apr_required, gain_v_time

def process_range(price_df, range_mode, r, fee_per_ut_per_tick, csv_name):
    if MANUAL:
        low_pct, high_pct = r
    else:
        high_pct, low_pct = get_high_low_pct(range_mode, r)
    high_tick, low_tick = get_ticks(high_pct, low_pct)
    gains, apr, gain_v_time = simulate_range(price_df, high_tick, low_tick, fee_per_ut_per_tick, False)
    result = {
        'simulation': f"{csv_name}",
        # 'gm_rebalance': False,
        'range': f"+{high_tick}/-{low_tick}",
        'gains': gains,
        'apr_needed': apr,
        'gain_v_time': gain_v_time
        }
    return result

def apply_weighting(result_df):
    result_df['second_word'] = result_df['simulation'].str.split('_').str[1]
    # Define the mapping dictionary
    weighting = {'high': 1, 'med': 17.5, 'low': 33}
    # Map the second word to the corresponding value and create a new column
    result_df['weighting'] = result_df['second_word'].map(weighting)
    result_df['weighted_gains'] = result_df['gains'] * result_df['weighting']
    out_df = result_df.pivot_table(index='range', values=['weighted_gains'], aggfunc=['sum'])
    out_df = out_df.sort_values([('sum', 'weighted_gains')], ascending=False)
    print(out_df)
    return out_df

def plot_lp_gains(df, title):
    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot the first column in the top subplot
    ax1.plot(df.index, df['price'], label='Price')
    ax1.set_title(title)
    ax1.set_ylabel('Price (USD)')
    ax1.legend()

    # Plot the remaining columns in the bottom subplot
    for column in df.columns[1:]:
        ax2.plot(df.index, df[column], label=column)

    ax2.set_title('LP Gain over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Gains (USD)')
    ax2.legend()

    # Adjust layout
    plt.tight_layout()

    # Save each plot based on the title
    plt.savefig(f"outputs/{title}.png")

    # Show the plot
    plt.show()

def combine_and_plot(result_df, csvs):
    for csv in csvs:
        df = pd.read_csv(csv, parse_dates=True, index_col='date')
        #df = df['2024-12-15':'2025-01-30']
        name = csv.split('/')[-1].split('.')[0]
        df = df['price']
        to_add = result_df.loc[result_df.loc[:, 'simulation'] == name].set_index('range').loc[:, 'gain_v_time']
        expanded = pd.DataFrame(to_add.to_list()).T
        expanded.columns = to_add.index
        expanded.index = df.index
        df = pd.concat([df, expanded], axis=1)
        plot_lp_gains(df, name)

def main():
    result_list = []
    # Get a list of csv files from the directory
    csvs = glob.glob(f"data/volatility_segments/{TOKEN0}*.csv")
    # csvs = ["data/binance_btc_minute.csv"]
    for csv in csvs:
        csv_name = csv.split("/")[-1].split(".")[0]
        price_df = pd.read_csv(csv, parse_dates=True, index_col='date')
        # price_df = price_df['2024-12-15':'2025-01-30']
        #plot_price(price_df)
        latest_price = price_df.loc[:, "price"].iloc[0]
        # print(f"\nSimulation Starting Price: ${latest_price:.2f}")
        fee_per_ut_per_tick = calculate_fee_per_ut_per_tick()

        with ProcessPoolExecutor() as executor:
            futures = []
            if MANUAL:
                for r in MANUAL_RANGES:
                    futures.append(executor.submit(process_range, price_df, RangeMode.MANUAL, r, fee_per_ut_per_tick, csv_name))
            else:
                for range_mode in [RangeMode.EVEN, RangeMode.FIXL, RangeMode.FIXH]:
                    for r in range(MIN_TOLERANCE, MAX_TOLERANCE + 1, 2):
                        futures.append(executor.submit(process_range, price_df, range_mode, r, fee_per_ut_per_tick, csv_name))
            for future in futures:
                result = future.result()  # Ensure all tasks are completed
                result_list.append(result)

        # print(f"Simulation Ending Price: ${price_df.loc[:, 'price'].iloc[-1]:.2f}")

    result_df = pd.DataFrame(result_list)
    # drop duplicates of simulation and range
    result_df = result_df.drop_duplicates(subset=['simulation', 'range'])
    print(result_df)
    # result_df.to_csv("outputs/sim_result.csv")
    # save the dataframe to a pickle file
    result_df.to_pickle("outputs/sim_result.pkl")
    combine_and_plot(result_df, csvs)
    # out_df = apply_weighting(result_df)
    print("Simulation Complete")

if __name__ == "__main__":
    main()