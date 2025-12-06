import private
import requests
import pandas as pd
import os
import core.helper_classes as hc
import json

class SickleNFTtracker:
    def __init__(self, start_block, end_block=99999999, wallet_addr=private.wal_lp, sickle_contract=private.sickle_lp, path_start="outputs"):
        self.start_block = start_block
        self.end_block = end_block
        self.wallet = wallet_addr.lower()
        self.wal_shortname = f"{self.wallet[2:6].lower()}-{self.wallet[-4:].lower()}"
        self.sickle = sickle_contract.lower()
        self.addr_filters = [self.wallet, self.sickle]
        # Use Alchemy instead of Etherscan
        self.alchemy_url = private.alchemy_base_url
        # Cache for token info
        self._token_info_cache = {}
        self.method_id_map = {
            "0x2812d614": "Compound",
            "0xf5304377": "Deposit",
            "0xee72c09a": "Deposit",
            "0x5ec5999e": "Harvest",
            "0x1d06722b": "Harvest",  # Simple Harvest
            "0xe6fb317f": "Rebalance",  # Normally RebalanceFor
            "0xe5bacdd0": "Increase",
            "0x1c396db6": "Decrease",
            "0x28734381": "Exit",
            "0x9b5c1ee5": "Exit",
            "0x659b91b1": "Rebalance",  # Normally Rebalance
            "0x22451262": "Rebalance",   # Normally Move
            "0xddff9470": "Deposit",
            "0x23717967": "Rebalance",  # Normally Move
            "0x8896bd16": "Rebalance"  # Normally Move
        }
        self.cgids = {
            'cbBTC': 'bitcoin',
            'WETH': 'ethereum',
            'AERO': 'aerodrome-finance',
            'USDC': 'usd-coin'
        }
        # topic id is needed to understand methods related to compound, harvest, etc.
        self.topic_id = "0xbf9d03ac543e8f596c6f4af5ab5e75f366a57d2d6c28d2ff9c024bd3f88e8771"
        self.csv_path = f"{path_start}/{self.wal_shortname}_tracker.csv"
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
            # if the last block seen has an eventType of "Exit", then continue
            if df_old[df_old['seriesID'] == seriesID]['eventType'].iloc[-1] == "Exit":
                print(f"{seriesID} was closed, continuing...")
                continue
            last_block_seen.append(df_old[
                (df_old["seriesID"] == seriesID) & 
                (df_old["eventType"] != "Harvest")
                ]["blockNumber"].max())
            print(f"Last block seen for seriesID {seriesID} is {last_block_seen[-1]}")  
        if not last_block_seen:
            print("No need to pull historical data")
            self.df_old = df_old
            return
        if self.end_block <  min(last_block_seen):
            print(f"extracting older data, keeping block pull from {self.start_block} to {self.end_block}")
        else:
            print(f"Data exisits, setting start block to {min(last_block_seen) - 1} in order to continue tracing NFT transfers")
            self.start_block = min(last_block_seen) - 1
        self.df_old = df_old
    
    def _get_token_info(self, token_address):
        """Get token symbol and decimals using Alchemy Token API, with caching"""
        token_address_lower = token_address.lower()
        if token_address_lower in self._token_info_cache:
            return self._token_info_cache[token_address_lower]
        
        try:
            # Use Alchemy's Token API instead of contract calls
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "alchemy_getTokenMetadata",
                "params": [token_address]
            }
            
            response = requests.post(self.alchemy_url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            data = response.json()
            
            if "result" in data and data["result"]:
                result = data["result"]
                info = {
                    'symbol': result.get('symbol', 'UNKNOWN'),
                    'decimals': result.get('decimals', 18)
                }
                self._token_info_cache[token_address_lower] = info
                return info
            else:
                return {'symbol': 'UNKNOWN', 'decimals': 18}
        except Exception as e:
            # Return default values if API call fails
            return {'symbol': 'UNKNOWN', 'decimals': 18}
    
    def fetch_raw_transactions(self, address, action='tokentx'):
        """
        Uses Alchemy API to fetch transactions for a given address
        For the NFT tracker, the address to use is your personal sickle contract one so it only pulls transactions for your NFT's
        Returns data in the same format as Etherscan API
        """
        is_nft = (action == 'tokennfttx')
        
        # Convert block numbers to hex for Alchemy
        from_block = hex(self.start_block) if isinstance(self.start_block, int) else self.start_block
        to_block = hex(self.end_block) if isinstance(self.end_block, int) and self.end_block < 99999999 else 'latest'
        
        # Use Alchemy's getAssetTransfers API
        all_transfers = []
        
        if is_nft:
            # For NFTs, query ERC721 transfers involving the address (same as Etherscan did)
            # Etherscan's tokennfttx action returned all NFT transfers for the address, including mints
            params_list = [
                {"fromAddress": address, "category": ["erc721"], "fromBlock": from_block, "toBlock": to_block, "withMetadata": True, "excludeZeroValue": False, "maxCount": "0x3e8"},
                {"toAddress": address, "category": ["erc721"], "fromBlock": from_block, "toBlock": to_block, "withMetadata": True, "excludeZeroValue": False, "maxCount": "0x3e8"}
            ]
        else:
            # For tokens, query ERC20 transfers only
            # Note: Internal transactions (native ETH) are not supported on Base network by Alchemy
            # ETH and WETH are treated the same - WETH is just wrapped ETH
            params_list = [
                {"fromAddress": address, "category": ["erc20"], "fromBlock": from_block, "toBlock": to_block, "withMetadata": True, "excludeZeroValue": False, "maxCount": "0x3e8"},
                {"toAddress": address, "category": ["erc20"], "fromBlock": from_block, "toBlock": to_block, "withMetadata": True, "excludeZeroValue": False, "maxCount": "0x3e8"}
            ]
        
        # Fetch with pagination
        for params in params_list:
            page_key = None
            while True:
                if page_key:
                    params['pageKey'] = page_key
                
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "alchemy_getAssetTransfers",
                    "params": [params]
                }
                
                response = requests.post(self.alchemy_url, json=payload, headers={"Content-Type": "application/json"})
                response.raise_for_status()
                data = response.json()
                
                if 'error' in data:
                    print(f"Error fetching data: {data['error']}")
                    break
                
                result = data.get('result', {})
                transfers = result.get('transfers', [])
                all_transfers.extend(transfers)
                
                page_key = result.get('pageKey')
                if not page_key:
                    break
        
        # Deduplicate
        seen = set()
        unique_transfers = []
        for transfer in all_transfers:
            tx_hash = transfer.get('hash', '').lower()
            if is_nft:
                # For NFTs, include from/to in key to distinguish mints from transfers
                # (same hash + tokenID can have multiple transfers: mint, transfer to farm, etc.)
                from_addr = transfer.get('from', '').lower()
                to_addr = transfer.get('to', '').lower()
                token_id = str(transfer.get('erc721TokenId') or transfer.get('tokenId', ''))
                key = (tx_hash, from_addr, to_addr, token_id)
            else:
                from_addr = transfer.get('from', '').lower()
                to_addr = transfer.get('to', '').lower()
                value = transfer.get('rawContract', {}).get('value', '')
                key = (tx_hash, from_addr, to_addr, str(value))
            
            if key not in seen:
                seen.add(key)
                unique_transfers.append(transfer)
        
        # Convert Alchemy format to Etherscan format
        results = []
        for transfer in unique_transfers:
            # Extract block number
            block_num_hex = transfer.get('blockNum', '0x0')
            if isinstance(block_num_hex, str):
                block_num = int(block_num_hex, 16) if block_num_hex.startswith('0x') else int(block_num_hex)
            else:
                block_num = int(block_num_hex)
            
            # Extract timestamp
            metadata = transfer.get('metadata', {})
            timestamp = None
            if metadata and isinstance(metadata, dict):
                timestamp_str = metadata.get('blockTimestamp')
                if timestamp_str:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        timestamp = int(dt.timestamp())
                    except:
                        pass
            
            tx_hash = transfer.get('hash', '').lower()  # Normalize hash to lowercase for consistent merging
            from_addr = transfer.get('from', '').lower()
            to_addr = transfer.get('to', '').lower()
            
            if is_nft:
                # NFT transfer - convert to Etherscan format
                # Alchemy provides tokenId in various formats - handle all cases
                token_id_raw = transfer.get('erc721TokenId') or transfer.get('tokenId', '')
                
                # Convert to integer
                if token_id_raw is None:
                    token_id = 0
                elif isinstance(token_id_raw, (int, float)):
                    token_id = int(token_id_raw)
                elif isinstance(token_id_raw, str):
                    token_id_str = token_id_raw.strip()
                    if not token_id_str or token_id_str == '0x' or token_id_str == '0x0':
                        token_id = 0
                    elif token_id_str.startswith('0x'):
                        try:
                            token_id = int(token_id_str, 16)
                        except ValueError:
                            token_id = 0
                    else:
                        try:
                            token_id = int(token_id_str)
                        except ValueError:
                            token_id = 0
                else:
                    token_id = 0
                
                results.append({
                    'blockNumber': str(block_num),
                    'timeStamp': str(timestamp) if timestamp else '0',
                    'hash': tx_hash,
                    'from': from_addr,
                    'to': to_addr,
                    'tokenID': token_id  # Keep as integer
                })
            else:
                # ERC20 transfer - convert to Etherscan format
                raw_contract = transfer.get('rawContract', {})
                value_hex = raw_contract.get('value', '0x0')
                if isinstance(value_hex, str) and value_hex.startswith('0x'):
                    value = str(int(value_hex, 16))
                else:
                    value = str(int(value_hex)) if value_hex else '0'
                
                contract_addr = raw_contract.get('address', '').lower()
                # Get token symbol and decimals from contract
                token_info = self._get_token_info(contract_addr)
                
                results.append({
                    'blockNumber': str(block_num),
                    'timeStamp': str(timestamp) if timestamp else '0',
                    'hash': tx_hash,
                    'from': from_addr,
                    'to': to_addr,
                    'value': value,
                    'tokenSymbol': token_info['symbol'],
                    'tokenDecimal': str(token_info['decimals'])
                })
        
        return results
    
    def fetch_transfers_for_hash(self, tx_hash):
        """
        Fetch all ERC20 transfers for a specific transaction hash.
        Used to get full transaction details for Deposit/Exit transactions.
        """
        # Normalize hash to lowercase
        tx_hash = tx_hash.lower() if isinstance(tx_hash, str) else tx_hash
        # Get the transaction to find its block number
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_getTransactionByHash",
            "params": [tx_hash]
        }
        
        response = requests.post(self.alchemy_url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        data = response.json()
        
        if "result" not in data or not data["result"]:
            return []
        
        block_num_hex = data["result"].get("blockNumber")
        if not block_num_hex:
            return []
        
        block_num = int(block_num_hex, 16) if isinstance(block_num_hex, str) else int(block_num_hex)
        
        # Query all ERC20 transfers for this specific block
        params = {
            "fromBlock": hex(block_num),
            "toBlock": hex(block_num),
            "category": ["erc20"],
            "withMetadata": True,
            "excludeZeroValue": False,
            "maxCount": "0x3e8"
        }
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "alchemy_getAssetTransfers",
            "params": [params]
        }
        
        response = requests.post(self.alchemy_url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data:
            return []
        
        result = data.get('result', {})
        transfers = result.get('transfers', [])
        
        # Filter to only transfers in this specific transaction
        tx_transfers = [t for t in transfers if t.get('hash', '').lower() == tx_hash.lower()]  # tx_hash already normalized above
        
        # Convert to our format
        results = []
        for transfer in tx_transfers:
            block_num_hex = transfer.get('blockNum', '0x0')
            if isinstance(block_num_hex, str):
                block_num = int(block_num_hex, 16) if block_num_hex.startswith('0x') else int(block_num_hex)
            else:
                block_num = int(block_num_hex)
            
            metadata = transfer.get('metadata', {})
            timestamp = None
            if metadata and isinstance(metadata, dict):
                timestamp_str = metadata.get('blockTimestamp')
                if timestamp_str:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        timestamp = int(dt.timestamp())
                    except:
                        pass
            
            tx_hash_transfer = transfer.get('hash', '').lower()  # Normalize hash to lowercase
            from_addr = transfer.get('from', '').lower()
            to_addr = transfer.get('to', '').lower()
            
            raw_contract = transfer.get('rawContract', {})
            value_hex = raw_contract.get('value', '0x0')
            if isinstance(value_hex, str) and value_hex.startswith('0x'):
                value = str(int(value_hex, 16))
            else:
                value = str(int(value_hex)) if value_hex else '0'
            
            contract_addr = raw_contract.get('address', '').lower()
            token_info = self._get_token_info(contract_addr)
            
            results.append({
                'blockNumber': str(block_num),
                'timeStamp': str(timestamp) if timestamp else '0',
                'hash': tx_hash_transfer,
                'from': from_addr,
                'to': to_addr,
                'value': value,
                'tokenSymbol': token_info['symbol'],
                'tokenDecimal': str(token_info['decimals'])
            })
        
        return results

    def fetch_transaction_details(self, tx_hash):
        """
        Fetches the details of a single transaction.  This is needed to get the method ID that was
        used in the transaction.  Returns the method name and tokenID.
        Note: tokenID should come from NFT transfers, not from parsing transaction input.
        This method only extracts method name from input data.
        """
        # Normalize hash to lowercase
        tx_hash = tx_hash.lower() if isinstance(tx_hash, str) else tx_hash
        # Use Alchemy's eth_getTransactionByHash
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_getTransactionByHash",
            "params": [tx_hash]
        }
        
        response = requests.post(self.alchemy_url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        data = response.json()
        
        if "result" in data and data["result"] and "input" in data["result"]:
            method_id = data["result"]["input"][:10]  # First 10 chars are the method ID
            method_name = self.method_id_map.get(method_id.lower(), "Unknown")
            print(f"{method_name} occured at tx {tx_hash[:6]}...{tx_hash[-4:]}")
            # Don't try to extract tokenID from input - it's unreliable and position varies by method
            # TokenID should come from NFT transfers instead
            return method_name, 0
        return "Unknown", 0
    
    def process_nft_transactions(self, nft_txns, token_txns):
        df = pd.DataFrame(nft_txns)
        df = df[["blockNumber", "timeStamp", "hash", "from", "to", "tokenID"]]
        df["hash"] = df["hash"].str.lower()  # Normalize hash to lowercase
        df["timeStamp"] = pd.to_datetime(pd.to_numeric(df["timeStamp"], errors='coerce'), unit='s')
        df.loc[:, "tokenID"] = df["tokenID"].astype(int)
        df = df[df['tokenID'] != 0]  # Remove SPAM NFT transactions
        
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
            if row["from"].lower() == self.wallet.lower():
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
        df.loc[df["from"].str.lower() == self.wallet.lower(), "amount"] *= -1  # Funds sent to contract/NFT are negative

        # Filter to only wallet<->contract transfers (user interactions - taxable events)
        df = df[df["from"].isin(self.addr_filters) & df["to"].isin(self.addr_filters)]
        
        # Normalize hashes to lowercase for consistent merging
        df['hash'] = df['hash'].str.lower()
        if self.nft_df is not None and len(self.nft_df) > 0:
            self.nft_df['hash'] = self.nft_df['hash'].str.lower()
            # Ensure required columns exist in nft_df
            required_cols = ['hash', 'seriesID', 'eventType', 'tokenID']
            available_cols = [col for col in required_cols if col in self.nft_df.columns]
            if len(available_cols) == len(required_cols):
                # Link token transfers to NFT burn and mint transactions
                df = df.merge(self.nft_df[available_cols], on='hash', how='left')
            else:
                # If nft_df doesn't have required columns, initialize them
                for col in ['seriesID', 'eventType', 'tokenID']:
                    if col not in df.columns:
                        df[col] = pd.NA
        else:
            # If nft_df is empty, initialize the columns
            for col in ['seriesID', 'eventType', 'tokenID']:
                if col not in df.columns:
                    df[col] = pd.NA
        
        # For rows still missing seriesID/tokenID, try to get them from nft_df by hash
        # (This handles cases where the merge didn't find a match)
        missing_mask = df['seriesID'].isna() | df['tokenID'].isna()
        if missing_mask.any():
            # Create a lookup from nft_df: hash -> (seriesID, tokenID)
            nft_lookup = self.nft_df.set_index('hash')[['seriesID', 'tokenID']].to_dict('index')
            for idx in df[missing_mask].index:
                tx_hash = df.loc[idx, 'hash']
                if tx_hash in nft_lookup:
                    if pd.isna(df.loc[idx, 'seriesID']):
                        df.loc[idx, 'seriesID'] = nft_lookup[tx_hash]['seriesID']
                    if pd.isna(df.loc[idx, 'tokenID']):
                        df.loc[idx, 'tokenID'] = nft_lookup[tx_hash]['tokenID']
        
        # Get a list of hashes that don't have an event type yet
        tx_hashes = df[pd.isna(df["eventType"])]["hash"].unique()
        # Fetch the event type for each hash (tokenID should already be from NFT transfers)
        event_types = {}
        for hash in tx_hashes:
            event_type, _ = self.fetch_transaction_details(hash)
            event_types[hash] = event_type

        # Apply the event type to the data frame by matching the hash value
        df.loc[df["hash"].isin(event_types.keys()), "eventType"] = df["hash"].map(event_types)
        
        # Targeted fix: For Deposit and Exit transactions ONLY, fetch full transaction details
        # to check for missing WETH transfers
        deposit_hashes = df[df["eventType"] == "Deposit"]["hash"].unique()
        exit_hashes = df[df["eventType"] == "Exit"]["hash"].unique()
        weth_to_add = []
        
        if len(deposit_hashes) > 0:
            # For each Deposit transaction, fetch ALL transfers in that transaction
            for tx_hash in deposit_hashes:
                all_tx_transfers = self.fetch_transfers_for_hash(tx_hash)
                df_tx = pd.DataFrame(all_tx_transfers)
                if len(df_tx) > 0:
                    df_tx["timeStamp"] = pd.to_datetime(pd.to_numeric(df_tx["timeStamp"], errors='coerce'), unit='s')
                    df_tx["value"] = (df_tx["value"].astype(float) / (10 ** df_tx["tokenDecimal"].astype(int))).round(7)
                    df_tx.rename(columns={'value': 'amount'}, inplace=True)
                    
                    # Look for contract -> pool WETH transfers in this transaction
                    contract_to_pool_weth = df_tx[
                        (df_tx['from'] == self.sickle) &
                        (~df_tx['to'].isin(self.addr_filters)) &  # To pool
                        (df_tx['tokenSymbol'] == 'WETH')
                    ]
                    
                    # Check if we already have WETH in wallet->contract for this hash
                    has_weth = df[
                        (df['hash'] == tx_hash) &
                        (df['tokenSymbol'] == 'WETH') &
                        (df['from'] == self.wallet) &
                        (df['to'] == self.sickle)
                    ]
                    
                    if len(has_weth) == 0 and len(contract_to_pool_weth) > 0:
                        # Missing WETH - add it as wallet->contract (negative amount for deposit)
                        weth_row = contract_to_pool_weth.iloc[0].copy()
                        weth_row['from'] = self.wallet
                        weth_row['to'] = self.sickle
                        weth_row['amount'] = -abs(weth_row['amount'])  # Negative for deposit
                        # Get eventType and other fields from df or nft_df
                        matching_rows = df[df['hash'] == tx_hash]
                        if len(matching_rows) > 0:
                            weth_row['eventType'] = matching_rows['eventType'].iloc[0] if not pd.isna(matching_rows['eventType'].iloc[0]) else 'Deposit'
                            # Try to get seriesID/tokenID from matching rows first
                            if 'seriesID' in matching_rows.columns and not pd.isna(matching_rows['seriesID'].iloc[0]):
                                weth_row['seriesID'] = matching_rows['seriesID'].iloc[0]
                            if 'tokenID' in matching_rows.columns and not pd.isna(matching_rows['tokenID'].iloc[0]):
                                weth_row['tokenID'] = matching_rows['tokenID'].iloc[0]
                        else:
                            weth_row['eventType'] = 'Deposit'
                        
                        # If still missing seriesID/tokenID, look up from nft_df
                        has_series_id = 'seriesID' in weth_row and not pd.isna(weth_row['seriesID'])
                        has_token_id = 'tokenID' in weth_row and not pd.isna(weth_row['tokenID'])
                        if not has_series_id or not has_token_id:
                            nft_match = self.nft_df[self.nft_df['hash'] == tx_hash]
                            if len(nft_match) > 0:
                                if not has_series_id:
                                    weth_row['seriesID'] = nft_match['seriesID'].iloc[0]
                                if not has_token_id:
                                    weth_row['tokenID'] = nft_match['tokenID'].iloc[0]
                            else:
                                if not has_series_id:
                                    weth_row['seriesID'] = pd.NA
                                if not has_token_id:
                                    weth_row['tokenID'] = pd.NA
                        weth_to_add.append(weth_row)
        
        if len(exit_hashes) > 0:
            # For each Exit transaction, fetch ALL transfers in that transaction
            for tx_hash in exit_hashes:
                all_tx_transfers = self.fetch_transfers_for_hash(tx_hash)
                df_tx = pd.DataFrame(all_tx_transfers)
                if len(df_tx) > 0:
                    df_tx["timeStamp"] = pd.to_datetime(pd.to_numeric(df_tx["timeStamp"], errors='coerce'), unit='s')
                    df_tx["value"] = (df_tx["value"].astype(float) / (10 ** df_tx["tokenDecimal"].astype(int))).round(7)
                    df_tx.rename(columns={'value': 'amount'}, inplace=True)
                    
                    # Look for contract -> wallet WETH transfers in this transaction
                    contract_to_wallet_weth = df_tx[
                        (df_tx['from'] == self.sickle) &
                        (df_tx['to'] == self.wallet) &  # To wallet
                        (df_tx['tokenSymbol'] == 'WETH')
                    ]
                    
                    # Check if we already have WETH in contract->wallet for this hash
                    has_weth = df[
                        (df['hash'] == tx_hash) &
                        (df['tokenSymbol'] == 'WETH') &
                        (df['from'] == self.sickle) &
                        (df['to'] == self.wallet)
                    ]
                    
                    if len(has_weth) == 0 and len(contract_to_wallet_weth) > 0:
                        # Missing WETH return - add it as contract->wallet (positive amount for exit)
                        weth_row = contract_to_wallet_weth.iloc[0].copy()
                        weth_row['amount'] = abs(weth_row['amount'])  # Positive for exit/withdrawal
                        # Get eventType and other fields from df or nft_df
                        matching_rows = df[df['hash'] == tx_hash]
                        if len(matching_rows) > 0:
                            weth_row['eventType'] = matching_rows['eventType'].iloc[0] if not pd.isna(matching_rows['eventType'].iloc[0]) else 'Exit'
                            # Try to get seriesID/tokenID from matching rows first
                            if 'seriesID' in matching_rows.columns and not pd.isna(matching_rows['seriesID'].iloc[0]):
                                weth_row['seriesID'] = matching_rows['seriesID'].iloc[0]
                            if 'tokenID' in matching_rows.columns and not pd.isna(matching_rows['tokenID'].iloc[0]):
                                weth_row['tokenID'] = matching_rows['tokenID'].iloc[0]
                        else:
                            weth_row['eventType'] = 'Exit'
                        
                        # If still missing seriesID/tokenID, look up from nft_df
                        has_series_id = 'seriesID' in weth_row and not pd.isna(weth_row['seriesID'])
                        has_token_id = 'tokenID' in weth_row and not pd.isna(weth_row['tokenID'])
                        if not has_series_id or not has_token_id:
                            nft_match = self.nft_df[self.nft_df['hash'] == tx_hash]
                            if len(nft_match) > 0:
                                if not has_series_id:
                                    weth_row['seriesID'] = nft_match['seriesID'].iloc[0]
                                if not has_token_id:
                                    weth_row['tokenID'] = nft_match['tokenID'].iloc[0]
                            else:
                                if not has_series_id:
                                    weth_row['seriesID'] = pd.NA
                                if not has_token_id:
                                    weth_row['tokenID'] = pd.NA
                        weth_to_add.append(weth_row)
        
        # Add missing WETH transfers
        if weth_to_add:
            df_weth = pd.DataFrame(weth_to_add)
            df = pd.concat([df, df_weth], ignore_index=True)
        
        # Final pass: Fill any remaining missing seriesID/tokenID from nft_df by hash
        missing_mask = df['seriesID'].isna() | df['tokenID'].isna()
        if missing_mask.any():
            nft_lookup = self.nft_df.set_index('hash')[['seriesID', 'tokenID']].to_dict('index')
            for idx in df[missing_mask].index:
                tx_hash = df.loc[idx, 'hash']
                if tx_hash in nft_lookup:
                    if pd.isna(df.loc[idx, 'seriesID']):
                        df.loc[idx, 'seriesID'] = nft_lookup[tx_hash]['seriesID']
                    if pd.isna(df.loc[idx, 'tokenID']):
                        df.loc[idx, 'tokenID'] = nft_lookup[tx_hash]['tokenID']
        
        # TokenID should already be populated from the NFT transfers merge above
        # Just ensure it's the right type
        if "tokenID" in df.columns:
            # Convert to int64 where possible
            df["tokenID"] = pd.to_numeric(df["tokenID"], errors='coerce').astype('Int64')

        # Fill missing seriesID using tokenID mapping (for cases where tokenID exists but seriesID doesn't)
        df.loc[df["seriesID"].isna() & df["tokenID"].notna(), "seriesID"] = df.loc[df["seriesID"].isna() & df["tokenID"].notna(), "tokenID"].map(self.series_map)

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
        # ensure start and end are at least 8 hours apart
        if end - start < pd.Timedelta(hours=8):
            start = start - pd.Timedelta(hours=8)
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
        #df.loc[df["tokenSymbol"] == "AERO", "transactionType"] = "fee"
        df.loc[df["eventType"] == "Harvest", "transactionType"] = "fee"
        df.loc[df["eventType"].isin(["Deposit", "Increase"]), "transactionType"] = "fund"
        df.loc[df["eventType"] == "Exit", "transactionType"] = "withdraw"
        df.loc[df["tokenSymbol"] == "AERO", "transactionType"] = "fee"
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
            df.drop_duplicates(subset=['seriesID', 'blockNumber', 'amount', 'tokenSymbol'], inplace=True, keep="first")
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
        
        # Query token transfers for both contract and wallet addresses
        # This ensures we capture wallet->contract transfers (from wallet query) 
        # and contract->wallet transfers (from contract query)
        token_txns_contract = self.fetch_raw_transactions(self.sickle, 'tokentx')
        token_txns_wallet = self.fetch_raw_transactions(self.wallet, 'tokentx')
        
        # Combine and deduplicate (same transfer might appear in both queries)
        all_token_txns = token_txns_contract + token_txns_wallet
        # Deduplicate by hash, from, to, value (same as in fetch_raw_transactions)
        seen = set()
        token_txns = []
        for tx in all_token_txns:
            key = (tx['hash'], tx['from'], tx['to'], tx['value'])
            if key not in seen:
                seen.add(key)
                token_txns.append(tx)
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
