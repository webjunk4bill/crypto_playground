import time

import cl_classes as cl
import pickle
import pandas as pd
import numpy as np
import os

"""
This simulation can only run on an hourly basis, therefore it favors very tight liquidity range because they always
get put in range before the fees are collected and there is only one gas fee paid.  Tight ranges could be changing on a 
minute by minute basis.  Not sure how to account for that.
"""

# Set up the time range that you want to download the historical in_data for
# Once downloaded, information is stored in a pickle file for easy testing and recovery (downloads can take some time)
download = False

start = pd.Timestamp('2024-05-01')
stop = pd.Timestamp('2024-06-18')

if download:
    # Set up LPs
    brett_weth_p3 = cl.V3DexLP('0x76Bf0abD20f1e0155Ce40A62615a90A709a6C3D8', 'base',
                               'based-brett', start, stop, 'eth')
    brett_weth_1p0 = cl.V3DexLP('0xBA3F945812a83471d709BCe9C3CA699A19FB46f7', 'base',
                                'based-brett', start, stop, 'eth')
    spec_weth_1p0 = cl.V3DexLP('0x8055e6de251e414e8393b20AdAb096AfB3cF8399', 'base',
                               'spectral', start, stop, 'eth')
    spec_weth_p3 = cl.V3DexLP('0xee6f3C5f418d1097c50C4698d535EDB33Bd72931', 'base',
                              'spectral', start, stop, 'eth')
    print("Waiting 2 mintues before executing next queries")
    time.sleep(120)
    aero_weth_p3 = cl.V3DexLP('0x3d5D143381916280ff91407FeBEB52f2b60f33Cf', 'base',
                              'aerodrome-finance', start, stop, 'eth')
    toshi_weth_p3 = cl.V3DexLP('0x5aa4AD647580bfE86258d300Bc9852F4434E2c61', 'base',
                               'toshi', start, stop, 'eth')
    eth_usdc_p05 = cl.V3DexLP('0xd0b53D9277642d899DF5C87A3966A349A798F224', 'base',
                              'ethereum', start, stop, 'usd')


def load_all_pickles_from_directory(directory):
    pickle_files = [f for f in os.listdir(directory) if f.endswith('.pkl') or f.endswith('.pickle')]
    loaded_objects = []
    for name in pickle_files:
        filepath = os.path.join(directory, name)
        try:
            obj = cl.V3DexLP.load_from_pickle(filepath)
            loaded_objects.append(obj)
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")
    return loaded_objects


def generate_parameter_combos(low_range, high_range, rebalance, autocompound):
    for a in low_range:
        for b in high_range:
            for c in rebalance:
                for d in autocompound:
                    yield a, b, c, d


def compute_outputs(a, b, c, d, investment, my_lp):
    result = my_lp.simulate_range(investment, a, b, c, d)
    return list(result.values())


def generate_output_values(low_range, high_range, rebalance, autocompound, investment, my_lp):
    for inputs in generate_parameter_combos(low_range, high_range, rebalance, autocompound):
        yield compute_outputs(*inputs, investment, my_lp)


# Set up range for analysis
investment = 1000
low_range = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
high_range = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
autocompound = [False, True]
rebalance = [False, True]

list_of_lps = load_all_pickles_from_directory('in_data')

for lp in list_of_lps:
    my_lp = cl.LiquidityPosition(lp)
    # DataFrame
    columns = ['low_range', 'high_range', 'rebalance', 'compound',
               '$Fees Accrued', '$Fees Compounded', '$Final Position', 'Rebalnces', '$Total Profit',
               '%Fee APR', '%Total APR']
    df = pd.DataFrame(columns=columns)

    # Loop
    for i, (x_in, y_out) in enumerate(zip(generate_parameter_combos(low_range, high_range, rebalance, autocompound),
                                          generate_output_values(low_range, high_range, rebalance, autocompound, investment,
                                                                 my_lp))):
        df.loc[i] = [*x_in, *y_out]

    df = df.sort_values(by='$Total Profit', ascending=False)
    filename = f"{lp.base_token.name}_{lp.quote_token.name}_{lp.fee_tier}.csv"
    f = open(f"out_data/{filename}", 'w')
    df.to_csv(f)
    f.close()
    print(df)
