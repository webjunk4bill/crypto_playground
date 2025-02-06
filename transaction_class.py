import private
import requests
from web3 import Web3
import json
import pandas as pd
import helper_classes as hc

class Transactions:
    def __init__(self, wallet):
        self.wallet = wallet
        self.wal_shortname = f"{wallet[2:6]}...{wallet[-4:]}"
        self.base_url = 'https://api.etherscan.io/v2/api'
        self.api_key = private.etherscan_api
        self.w3 = Web3(Web3.HTTPProvider(private.grove_base_url))
        self.transaction_list = []
        self.filtered_list = None
        # Define Method IDs for filtering
        self.method_id = {
            'rebalanceFor': 'e6fb317f',
            'harvest': '5ec5999e'
            }
        self.tokens = {
            'USDC': '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913',
            'WETH': '0x4200000000000000000000000000000000000006',
            'cbBTC': '0xcbB7C0000aB88B473b1f5aFd9ef808440eed33Bf',
            'AERO': '0x940181a94A35A4569E4529A3CDfB74e38FD98631'
        }
        self.cgids = {
            'cbBTC': 'bitcoin',
            'WETH': 'ethereum',
            'AERO': 'aerodrome-finance',
            'USDC': 'usd-coin'
        }
        self.token_prices = {}
        self.logs = None
    # TODO: Make sure to have a save a load function from csv

    def get_transactions(self, start_block):
        try:
            end_block = self.w3.eth.get_block_number()
        except Exception as e:
            print(f"Error occurred while getting end_block: {e}")
            raise
        # Just pull for given token IDs (based on Contract Address)
        for name, contract_address in self.tokens.items():
            params = {
                'chainid': '8453',  # base mainnet = 8453
                'module': 'account',
                'action': 'tokentx',
                'contractaddress': contract_address,  # Only need to filter by token contract
                'address': self.wallet,
                'page': '1',
                'offset': '100',
                'startblock': start_block,
                'endblock': end_block,
                'sort': 'asc',
                'apikey': self.api_key
            }

            response = requests.get(f'{self.base_url}', params=params)
            try:
                response.raise_for_status()
                print(f"Received transactions for {name}")
            except requests.exceptions.HTTPError as e:
                print(f"Error occurred: {e}")
                print(f"Status code: {response.status_code}")
            data = json.loads(response.text)
            self.transaction_list.extend(data['result'])
        return
    
    def get_method_id(self, tx_hash):
        """Fetch the transaction input and extract the method ID (first 4 bytes)."""
        tx = self.w3.eth.get_transaction(tx_hash)
        input_data = tx["input"][:10].hex()
        return input_data[:8] if input_data else None  # First 2 bytes as string
    
    def filter_tx_list_by_method(self):
        """Filter the transaction list by method ID."""
        filtered_tx_list = []
        for tx in self.transaction_list:
            tx_hash = tx['hash']
            method_id = self.get_method_id(tx_hash)
            if method_id in self.method_id.values():
                filtered_tx_list.append(tx)
                continue
        return filtered_tx_list
    
    def filter_tx_list_by_sender(self, sender):
        """Filter the transaction list by sender address."""
        filtered_tx_list = []
        for tx in self.transaction_list:
            tx_sender = tx['from'].lower()
            if tx_sender == sender.lower():
                filtered_tx_list.append(tx)
                continue
        self.filtered_list = filtered_tx_list
        print(f"Received {len(filtered_tx_list)} transactions for sender: {sender}")
        return filtered_tx_list
    
    def create_dataframe(self):
        """Create a pandas DataFrame from the filtered transaction list."""
        df = pd.DataFrame(self.filtered_list)
        col_to_keep = ['timeStamp', 'blockNumber', 'hash', 'value', 'tokenDecimal', 'tokenSymbol']
        df = df[col_to_keep]
        df['dateTime'] = pd.to_datetime(df['timeStamp'].astype(int), unit='s')
        df['value'] = df['value'].astype(float) / 10 ** df['tokenDecimal'].astype(int)
        df.rename(columns={"value": "amount"})
        df = df.assign(Recorded=False)
        df.sort_values(by='blockNumber', ascending=False, inplace=True)
        df.set_index('dateTime', inplace=True)
        self.df = df
        print(f"First Block Stored: {df.loc[:,'blockNumber'].min()}\nLast Block Stored: {df.loc[:,'blockNumber'].max()}")

    def get_usd_prices(self):
        """
        Get the USD prices for the tokens in the DataFrame.
        Returns a dictionary of pandas Series of prices for each token.
        """
        df = self.df
        token_symbols = df['tokenSymbol'].unique()
        start = df.index.min()
        end = df.index.max()
        prices = {}
        for symbol in token_symbols:
            try:
                cgid = self.cgids[symbol]
            except:
                print(f"Could not find cgid for {symbol}")
            p, v = hc.get_historical_prices(cgid, start, end)
            prices[symbol] = p
        self.token_prices = prices
        return prices
    
    def update_prices(self):
        """
        Update the prices in the DataFrame
        Calculate/add the total USD value of the transaction
        """
        self.df['price'] = self.df.apply(lambda row: self.find_closest_price(row['tokenSymbol'], row.name), axis=1)
        self.df['valueUsd'] = self.df['price'] * self.df['value']

    def find_closest_price(self, token, timestamp):
        """
        Find the closest price for a given token and timestamp.
        """
        if token in self.token_prices:
            series = self.token_prices[token]  # Retrieve the Pandas Series
            
            # Ensure the index is sorted (just in case)
            series = series.sort_index()
            
            # Find the position where 'timestamp' would be inserted
            pos = series.index.searchsorted(timestamp)

            # Handle edge cases where timestamp is outside the range
            if pos == 0:
                closest_time = series.index[0]
            elif pos >= len(series):
                closest_time = series.index[-1]
            else:
                # Compare the closest previous and next times
                prev_time = series.index[pos - 1]
                next_time = series.index[pos]
                closest_time = prev_time if abs(prev_time - timestamp) <= abs(next_time - timestamp) else next_time
            
            return series[closest_time]  # Return the price for the closest time

        return None  # If token not found

