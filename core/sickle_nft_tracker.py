import private
import requests
import pandas as pd
import os
import core.helper_classes as hc

class SickleNFTtracker:
    def __init__(self, start_block, end_block=99999999, wallet_addr=private.wal_lp, sickle_contract=private.sickle_lp):
        self.start_block = start_block
        self.end_block = end_block
        self.wallet = wallet_addr.lower()
        self.wal_shortname = f"{self.wallet[2:6].lower()}-{self.wallet[-4:].lower()}"
        self.sickle = sickle_contract.lower()
        self.addr_filters = [self.wallet, self.sickle]
        self.base_url = 'https://api.etherscan.io/v2/api'
        self.api_key = private.etherscan_api
        self.method_id_map = {
            "0x2812d614": "Compound",
            "0xf5304377": "Deposit",
            "0x5ec5999e": "Harvest",
            "0xe6fb317f": "Rebalance",  # Normally RebalanceFor
            "0xe5bacdd0": "Increase",
            "0x1c396db6": "Decrease",
            "0x28734381": "Exit",
            "0x659b91b1": "Rebalance",  # Normally Rebalance
            "0x22451262": "Rebalance"   # Normally Move
        }
        self.cgids = {
            'cbBTC': 'bitcoin',
            'WETH': 'ethereum',
            'AERO': 'aerodrome-finance',
            'USDC': 'usd-coin'
        }
        # topic id is needed to understand methods related to compound, harvest, etc.
        self.topic_id = "0xbf9d03ac543e8f596c6f4af5ab5e75f366a57d2d6c28d2ff9c024bd3f88e8771"
        self.csv_path = f"outputs/{self.wal_shortname}_tracker.csv"
        self.df_main = None
        self.df_old = None
        self.token_prices = None
        if os.path.exists(self.csv_path):
            self.read_stored_data()
        self.series_map = {}

    def read_stored_data(self):
        df_old = pd.read_csv(self.csv_path, parse_dates=["timeStamp"])
        df_old.set_index('timeStamp', inplace=True)
        df_old.sort_values(by='blockNumber', ascending=True, inplace=True)
        # Find the last block seen for each seriesID in the dataframe
        last_block_seen = []
        for seriesID in df_old['seriesID'].unique():
            last_block_seen.append(df_old[df_old['seriesID'] == seriesID]['blockNumber'].iloc[-2])
            print(f"Last block seen for seriesID {seriesID} is {last_block_seen[-1]}")  
        if self.end_block <  min(last_block_seen):
            print(f"extracting older data, keeping block pull from {self.start_block} to {self.end_block}")
        else:
            print(f"Data exisits, setting start block to {min(last_block_seen) - 1} in order to continue tracing NFT transfers")
            self.start_block = min(last_block_seen) - 1
        self.df_old = df_old
    
    def fetch_raw_transactions(self, address, action='tokentx'):
        """
        Uses the Etherscan API to fetch transactions for a given address
        For the NFT tracker, the address to use is your personal sickle contract one so it only pulls transactions for your NFT's
        """
        params = {
            'chainid': '8453',  # base mainnet = 8453
            'module': 'account',
            'action': action,
            # 'contractaddress': contract_address,
            'address': address,
            'page': '1',
            'offset': '10000',
            'startblock': self.start_block,
            'endblock': self.end_block,
            'sort': 'asc',
            'apikey': self.api_key
        }

        response = requests.get(f'{self.base_url}', params=params)
        data = response.json()
        if data["status"] != "1":
            print("Error fetching data:", data["message"], data['result'])
            return []
        
        return data["result"]

    def fetch_transaction_details(self, tx_hash):
        """
        Fetches the details of a single transaction.  This is needed to get the method ID that was
        used in the transaction.  It also retrieves the token ID (NFT ID), however right now it only
        works for methods where the NFT already exists, like Harvest and Compound
        """
        url = f"{self.base_url}?chainid=8453&module=proxy&action=eth_getTransactionByHash&txhash={tx_hash}&apikey={self.api_key}"
        response = requests.get(url)
        data = response.json()
        
        if "result" in data and "input" in data["result"]:
            method_id = data["result"]["input"][:10]  # First 10 chars are the method ID
            token_id_pos = 64 * 3 + 10
            token_id = int(data["result"]["input"][token_id_pos:token_id_pos + 64], 16)
            x = self.method_id_map.get(method_id.lower(), "Unknown")
            print(f"{x} occured at tx {tx_hash[:6]}...{tx_hash[-4:]}")
            return x, token_id
        return "Unknown"
    
    def process_nft_transactions(self, nft_txns, token_txns):
        df = pd.DataFrame(nft_txns)
        df = df[["blockNumber", "timeStamp", "hash", "from", "to", "tokenID"]]
        df["timeStamp"] = pd.to_datetime(pd.to_numeric(df["timeStamp"], errors='coerce'), unit='s')
        df["tokenID"] = df["tokenID"].astype(int)
        
        # Identify burns (to burn address) and mints (from address is zero address)
        burn_address = "0x0000000000000000000000000000000000000000"
        df = df[df["from"].eq(burn_address) | df["to"].eq(burn_address)]  # only care about mint and burn, not transfers to the Aerodrome farms
        df.loc[df["from"] == burn_address, "eventType"] = "Mint"
        df.loc[df["to"] == burn_address, "eventType"] = "Burn"
        
        # Identify NFT series
        df.loc[:, "seriesID"] = pd.NA  # Initialize so that can just fill na values
        # Add previous mapping values if they exist:
        if self.df_old is not None:
            self.series_map.update(self.df_old.set_index("tokenID")["seriesID"].to_dict())
        # Create new mapping for fresh deposits
        df_token = pd.DataFrame(token_txns)
        for _, row in df_token.iterrows():
            if row["from"].lower() == private.wal_lp.lower():
                tx_hash = row["hash"]
                # Extract rows from df_token with the same hash value
                subset = df_token[df_token["hash"] == tx_hash]
                tokens = subset["tokenSymbol"].tolist()
                # Create string out of the first two items in tokens list
                series_id = '/'.join(map(str, tokens[:2])) + '_' + tx_hash[-4:]
                matching_nft_mints = df[(df["hash"] == tx_hash) & (df["eventType"] == "Mint")]
                for _, mint in matching_nft_mints.iterrows():
                    self.series_map[mint["tokenID"]] = series_id
            
        # Assign series ID to subsequent burns and mints
        df.loc[df["seriesID"].isna(), "seriesID"] = df.loc[df["seriesID"].isna(), "tokenID"].map(self.series_map)
        for _, row in df.iterrows():
            if row["eventType"] == "Burn" and row["tokenID"] in self.series_map:
                tx_hash = row["hash"]
                new_mint = df[(df["hash"] == tx_hash) & (df["eventType"] == "Mint")]
                for _, mint in new_mint.iterrows():
                    self.series_map[mint["tokenID"]] = self.series_map[row["tokenID"]]
        df.loc[:, "seriesID"] = df["tokenID"].map(self.series_map)
        # Remove the Burns, keep only the Mints
        df = df.loc[df["eventType"] == "Mint"]

        # Now that Series IDs are mapped, update proper event Types
        tx_hashes = df["hash"].unique()
        event_type = {}
        for tx_hash in tx_hashes:
            event_type[tx_hash], _ = self.fetch_transaction_details(tx_hash)
        # map the event types for each tx_hash in df
        df.loc[:, "eventType"] = df["hash"].map(event_type)
        df.sort_values(by=["seriesID", "timeStamp"], inplace=True)
        self.nft_df = df
        return df
    
    def process_token_transfers(self, token_txns):
        """
        NFT transactions must be processed prior to processing the token transfers
        """
        df = pd.DataFrame(token_txns)
        df = df[["blockNumber", "timeStamp", "hash", "from", "to", "value", "tokenSymbol", "tokenDecimal"]]
        df["timeStamp"] = pd.to_datetime(pd.to_numeric(df["timeStamp"], errors='coerce'), unit='s')
        df["value"] = (df["value"].astype(float) / (10 ** df["tokenDecimal"].astype(int))).round(7)  # Convert to standard token value and round to 5 decimal digits
        df.rename(columns={'value': 'amount'}, inplace=True)  # Track as amount instead of value
        df.loc[df["from"].str.lower() == private.wal_lp.lower(), "amount"] *= -1  # Funds sent to contract/NFT are negative

        # Filter only transactions where both 'from' and 'to' are in FILTER_ADDRESSES
        df = df[df["from"].isin(self.addr_filters) & df["to"].isin(self.addr_filters)]
        # Link token transfers to NFT burn and mint transactions
        df = df.merge(self.nft_df[['hash', 'seriesID', 'eventType', 'tokenID']], on='hash', how='left')
        # Remove duplicate token transfers within the same transaction hash
        # df = df.drop_duplicates(subset=["hash", "from", "to", "amount", "tokenSymbol"])

        # Get a list of hashes that don't have an event type yet
        tx_hashes = df[pd.isna(df["eventType"])]["hash"].unique()
        # Fetch the event type and token ids for each hash
        event_types = {}
        token_ids = {}
        for hash in tx_hashes:
            event_types[hash], token_ids[hash] = self.fetch_transaction_details(hash)

        # Apply the event type and token ids to the data frame by matching the hash value
        df.loc[df["hash"].isin(event_types.keys()), "eventType"] = df["hash"].map(event_types)
        df.loc[df["hash"].isin(token_ids.keys()), "tokenID"] = df["hash"].map(token_ids)
        df["tokenID"] = df["tokenID"].astype(int)

        # Fill missing seriesID using tokenID mapping
        df.loc[df["seriesID"].isna(), "seriesID"] = df["tokenID"].map(self.series_map)

        df.sort_values(by=["seriesID", "timeStamp"], inplace=True)
        self.df_token = df
        return df
    
    def get_usd_prices(self):
        """
        Get the USD prices for the tokens in the DataFrame.
        Returns a dictionary of pandas Series of prices for each token.
        """
        token_symbols = self.df_main['tokenSymbol'].unique()
        start = self.df_main.index.min()
        end = self.df_main.index.max()
        prices = {}
        for symbol in token_symbols:
            try:
                cgid = self.cgids[symbol]
            except:
                print(f"Could not find cgid for {symbol}")
            p, v = hc.get_historical_prices(cgid, start, end)
            prices[symbol] = p
        self.token_prices = prices
        self.update_prices()
        return prices
    
    def update_prices(self):
        """
        Update the prices in the DataFrame
        Calculate/add the total USD value of the transaction
        """
        self.df_main['price'] = (self.df_main.apply(lambda row: self.find_closest_price(row['tokenSymbol'], row.name), axis=1)).round(3)
        self.df_main['valueUsd'] = (self.df_main['price'] * self.df_main['amount']).round(2)

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
    
    def update_transaction_type(self):
        """
        This only works accurately when the fees are being harvested in AERO during an automatic rebalance
        that includes harvesting.  With a manual harvest, it will still count it correctly regardless of the 
        token used.
        """
        df = self.df_main
        df["transactionType"] = "dust"
        df.loc[df["tokenSymbol"] == "AERO", "transactionType"] = "fee"
        df.loc[df["eventType"] == "Harvest", "transactionType"] = "fee"
        df.loc[df["eventType"].isin(["Deposit", "Increase"]), "transactionType"] = "fund"
        df.loc[df["eventType"] == "Exit", "transactionType"] = "withdraw"
        self.df_main = df
        return
    
    def merge_dataframes(self):
        if self.df_old is not None:
            print("Merging Dataframes")
            print(f"Historical Block range: {self.df_old.loc[:,'blockNumber'].min()} to {self.df_old.loc[:,'blockNumber'].max()}")
            print(f"Current Block range: {self.df_main.loc[:,'blockNumber'].min()} to {self.df_main.loc[:,'blockNumber'].max()}")
            df = pd.concat([self.df_old.copy(), self.df_main.copy()])
            df.loc[:, "blockNumber"] = df.loc[:, "blockNumber"].astype(int)
            df.sort_values(by=["seriesID", "blockNumber"], inplace=True)
            df.drop_duplicates(inplace=True, keep="last")
            # Any amount that is negative needs to have the eventType updated to Deposit
            df.loc[df["amount"] < 0, "eventType"] = "Deposit"
            print(f"Merged Block range: {df.loc[:,'blockNumber'].min()} to {df.loc[:,'blockNumber'].max()}")
            self.df_main = df
        else:
            self.df_main.loc[self.df_main["amount"] < 0, "eventType"] = "Deposit"

    def mark_transactions_recorded(self):
        self.df_main.loc[:,'recorded'] = True
    
    def read_and_process_transactions(self):
        """
        Reads the transactions from the Etherscan API and processes them. 
        The Sickle contract is what needs to be read in for proper filtering 
        NFT transactions need to be processed first, then fed into token transactions
        """
        nft_txns = self.fetch_raw_transactions(self.sickle, 'tokennfttx')
        if len(nft_txns) == 0:
            print("No NFT transactions found.  Continuing")
        token_txns = self.fetch_raw_transactions(self.sickle, 'tokentx')
        if len(token_txns) == 0:
            print("No Token transfer transactions found.  Checking old DF")
            if self.df_old is not None:
                print("Found old DF.  Copying to main df")
                self.df_main = self.df_old.copy()
                return
            else:
                print("No old DF found.  Aborting")
                return
        self.process_nft_transactions(nft_txns, token_txns)
        self.process_token_transfers(token_txns)
        # May want to do more processing, but for now assign the token df to main
        self.df_main = self.df_token
        self.df_main.set_index('timeStamp', inplace=True)
        # Get usd price data
        self.get_usd_prices()
        self.update_transaction_type()
        # Mark new transactions as NOT recorded
        self.df_main = self.df_main.assign(recorded=False)  # New transactions are not recorded
        self.merge_dataframes()
        return self.df_main.copy()
    
    def write_csv(self):
        self.df_main.to_csv(self.csv_path, date_format='%Y-%m-%d %H:%M:%S')
