import api
import requests
from web3 import Web3
import json
import pandas as pd
import binascii

class Transactions:
    def __init__(self, wallet):
        self.wallet = wallet
        self.base_url = 'https://api.etherscan.io/v2/api'
        self.api_key = api.etherscan_api
        self.w3 = Web3(Web3.HTTPProvider(api.grove_base_url))
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
        return filtered_tx_list
    
    def create_dataframe(self):
        """Create a pandas DataFrame from the filtered transaction list."""
        df = pd.DataFrame(self.filtered_list)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['value'] = df['value'].astype(float)
        df['gasPrice'] = df['gasPrice'].astype(float)
        df['gasUsed'] = df['gasUsed'].astype(float)
        df['gas'] = df['gasPrice'] * df['gasUsed']
        df['blockNumber'] = df['blockNumber'].astype(int)
        df['nonce'] = df['nonce'].astype(int)
        df['gasPriceGwei'] = df['gasPrice'] / 10**9
        df['gasUsedGwei'] = df['gasUsed'] / 10**9
        df['gasGwei'] = df['gasGwei'] / 10**9
        return df
    
    def get_tx_count(self):
        """Return the number of transactions in the filtered list."""
        return len(self.filtered_list)
    
    def get_total_value(self):
        """Return the total value of transactions in the filtered list."""
        return sum(self.filtered_list['value'])
    
    def get_total_gas(self):
        """Return the total gas cost of transactions in the filtered list."""
        return sum(self.filtered_list['gas'])
    
    def get_total_gas_used(self):
        """Return the total gas used of transactions in the filtered list."""
        return sum(self.filtered_list['gasUsed'])

