from web3 import Web3
import json
from api import base_url
import unimath as uni
import pandas as pd
import time
import csv


# not necessarily the ticks themselves.

def tick_to_word(tick, spacing):
    compressed = tick // spacing
    if tick < 0 and tick % spacing != 0:
        compressed -= 1
    return compressed >> 8


# Function to get the tick bitmap at a specific word position
def get_tick_bitmap(word_position):
    try:
        return pool_contract.functions.tickBitmap(word_position).call()
    except Exception as e:
        print(f"Error fetching tick bitmap at word position {word_position}: {e}")
        return 0


def get_tick_spacing():
    try:
        return pool_contract.functions.tickSpacing().call()
    except Exception as e:
        print(f"Error fetching tick spacing: {e}")
        return 0


def get_initialized_tick_indices(word_pos: int, price_range, spacing: int, compress=True):
    num_words = int((price_range / 0.0001) // 256 + 1)
    word_pos_indices = range(word_pos - num_words, word_pos + num_words + 1, 1)
    bitmaps = []
    for i in word_pos_indices:
        bitmaps.append(get_tick_bitmap(i))
    tick_indices = []
    for j in range(len(word_pos_indices)):
        ind = word_pos_indices[j]
        bitmap = bitmaps[j]
        if bitmap != 0:
            for i in range(256):
                bit = 1
                initialized = (bitmap & (bit << i)) != 0
                if initialized:
                    tick_index = (ind * 256 + i) * spacing
                    tick_indices.append(tick_index)
    if compress:
        ratio = len(tick_indices) // 100
        extractors = range(0, len(tick_indices), ratio)
        new_indices = [tick_indices[i] for i in extractors]
    else:
        new_indices = tick_indices

    return new_indices


def get_liquidity_per_tick(pool_name: str, indices):
    # This takes a very long time to query all of the ticks
    liquidity = {}
    missed_ticks = []
    with open(f"{pool_name}.csv", 'a', newline='') as csvfile:
        file = csv.writer(csvfile)
        file.writerow(['tick', 'liquidity'])
        i = 0
        while i < len(indices):
            try:
                result = pool_contract.functions.ticks(indices[i]).call()
                file.writerow([indices[i], result[0]])
                print(f"Liquidity at tick {indices[i]} is {result[0]}")
                time.sleep(0.25)
                i += 1
            except Exception as e:
                print(f"Error fetching info for tick {indices[i]}: {e}")
                print("wait 5 seconds and retry")
                time.sleep(5)
        csvfile.close()
    return liquidity


# Connect to your network
web3 = Web3(Web3.HTTPProvider(base_url))
pool_name = "eth_brett_0p3"
pool_address = "0x76Bf0abD20f1e0155Ce40A62615a90A709a6C3D8"
with open('univ3contract_abi.json', 'r') as abi_file:
    contract_abi = json.load(abi_file)
# Create the pool contract instance
pool_contract = web3.eth.contract(address=pool_address, abi=contract_abi)

# Fetch the current tick from slot0
tick_pct = 0.0001
slot0 = pool_contract.functions.slot0().call()
current_price_tick = slot0[1]
tick_spacing = get_tick_spacing()
current_liquidity_tick = current_price_tick // tick_spacing * tick_spacing
# Get 5 liquidity ticks above and 4 below the current one.  Go in order from middle out
tick_refs = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5]
ticks_to_get = []
for ref in tick_refs:
    ticks_to_get.append(ref * tick_spacing + current_liquidity_tick)

# word = tick_to_word(current_tick, tick_spacing)
# p_range = 0.2

# tick_indices = get_initialized_tick_indices(word, p_range, tick_spacing, False)
liquidity = get_liquidity_per_tick(f'{pool_name}', ticks_to_get)
