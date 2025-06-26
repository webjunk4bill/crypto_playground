import pandas as pd
from web3 import Web3
import json
import private
import unimath as um
import time

# Key Constants
LP_FACTORY_CONTRACT = '0x5e7BB104d84c7CB9B682AaC2F3d509f5F406809A'
SICKLE_NFT_CONTRACT = '0x827922686190790b37229fd06084350E74485b72'

def read_sickle_contract_position(token_id):
    web3 = Web3(Web3.HTTPProvider(private.grove_base_url))
    # Start with Sickle NFT contract
    info = get_nft_info(web3, token_id)
    decoded = {
        'nonce': info[0],
        'operator': info[1],
        'token0': info[2],
        'token1': info[3],
        'tickSpacing': info[4],
        'tickLower': info[5],
        'tickUpper': info[6],
        'liquidity': info[7],
        'feeGrowthInside0': info[8],
        'feeGrowthInside1': info[9],
        'tokensOwed0': info[10],
        'tokensOwed1': info[11],
        }
    # get token name and decimal usage
    print(f"Reading token information for token address {decoded['token0']}")
    decoded['token0Name'], decoded['token0Decimal'] = get_token_information(web3, decoded['token0'])
    print(f"Reading token information for token address {decoded['token1']}")
    decoded['token1Name'], decoded['token1Decimal'] = get_token_information(web3, decoded['token1'])
    # Get the pool address
    print(f"Reading pool information for token addresses {decoded['token0']} and {decoded['token1']}")
    decoded['poolAddress'] = get_pool_contract(web3, decoded['token0'], decoded['token1'], decoded['tickSpacing'])
    # get the tick value
    print(f"Reading current tick value for pool address {decoded['poolAddress']}")
    decoded["tickCurrent"] = get_current_tick(web3, decoded['poolAddress'])
    return decoded

def get_nft_info(web3, token_id, retries=5, delay=3):
    with open('abis/sickle_nft_abi.json', 'r') as nft_abi:
        sickle_nft_abi = json.load(nft_abi)
    sickle_contract = web3.eth.contract(address=SICKLE_NFT_CONTRACT, abi=sickle_nft_abi)
    """ Retrieves NFT position data with retry logic. """
    for attempt in range(retries):
        try:
            print(f"Reading NFT position for NFT ID {token_id} (Attempt {attempt + 1})")
            info = sickle_contract.functions.positions(token_id).call()
            return info
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}\nNFT ID {token_id} may not exist anymore.")
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
            else:
                print(f"Max retry attempts reached for NFT ID {token_id}. Returning None.")
                info = None
                return info
    
def get_token_information(web3, token_addr, retries=5, delay=3):
    with open('abis/token_abi.json', 'r') as token_abi_file:
        token_abi = json.load(token_abi_file)

    token_contract = web3.eth.contract(address=token_addr, abi=token_abi)

    def fetch_with_retries(func_name):
        """ Helper function to retry contract calls """
        for attempt in range(retries):
            try:
                return getattr(token_contract.functions, func_name)().call()
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: Could not get {func_name} for token {token_addr} due to {e}")
                if attempt < retries - 1:
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
                else:
                    print(f"Max retry attempts reached for {func_name}. Returning None.")
                    return None

    decimal = fetch_with_retries("decimals")
    name = fetch_with_retries("symbol")

    return name, decimal

def get_pool_contract(web3, token0, token1, tick_spacing, retries=5, delay=3):
    with open('abis/lp_factory_abi.json', 'r') as abi_file:
        abi = json.load(abi_file)
    
    contract = web3.eth.contract(address=LP_FACTORY_CONTRACT, abi=abi)
    
    for attempt in range(retries):
        try:
            pool_addr = contract.functions.getPool(token0, token1, tick_spacing).call()
            return pool_addr
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: Could not get pool address for {token0}/{token1} due to {e}")
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
            else:
                print("Max retry attempts reached. Returning None.")
                return None

def get_current_tick(web3, address, retries=5, delay=3):
    with open('abis/sickle_pool_abi.json', 'r') as abi_file:
        abi = json.load(abi_file)

    contract = web3.eth.contract(address=address, abi=abi)

    for attempt in range(retries):
        try:
            slot0 = contract.functions.slot0().call()
            return slot0[1]  # The current tick value
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: Could not get current tick for pool {address} due to {e}")
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
            else:
                print("Max retry attempts reached. Returning None.")
                return None


class SickleNFTcalculator:
    """
    Class to read in the csv output from the tracker class and perform various
    and maipulations of the data
    """
    def __init__(self, wallet_addr, path_start="outputs"):
        self.wallet = wallet_addr.lower()
        self.wal_shortname = f"{self.wallet[2:6].lower()}-{self.wallet[-4:].lower()}"
        self.csv_path = f"{path_start}/{self.wal_shortname}_tracker.csv"
        self.df = pd.read_csv(self.csv_path, parse_dates=["timeStamp"]).set_index("timeStamp")
        self.df_per_nft = self.get_each_nft_seriesID()
        
    def get_each_nft_seriesID(self):
        """
        Returns a dictionary of each NFT and the seriesID
        """
        # create mulitple dataframes, one for each NFT (seriesID)
        seriesID_dict = {}
        for seriesID in self.df.seriesID.unique():
            seriesID_df = self.df[self.df.seriesID == seriesID]
            seriesID_dict[seriesID] = seriesID_df
        return seriesID_dict
    
    def get_daily_fees(self, recorded="all"):
        """
        Retrieve a dataframe of the daily fees
        """
        if "date" not in self.df.columns:
            self.df["date"] = self.df.index.date
        if recorded == "new":
            piv_df = self.df[(self.df["transactionType"] == "fee") & (self.df["recorded"] == False)].pivot_table(index=["date", "tokenSymbol"], values=["amount", "valueUsd"], aggfunc='sum')
            if piv_df.empty:
                print("No new fees recorded")
                return None
        else:
            piv_df = self.df[self.df["transactionType"] == "fee"].pivot_table(index=["date", "tokenSymbol"], values=["amount", "valueUsd"], aggfunc='sum')
        piv_df = piv_df.reset_index()
        piv_df["date"] = pd.to_datetime(piv_df["date"])
        piv_df = piv_df.sort_values(by=["date"])
        total_fees = piv_df["valueUsd"].sum()
        if recorded == "new":
            print(f"Total Fees collected since last update: ${total_fees:.2f}")
            print(piv_df.pivot_table(index=["tokenSymbol"], values=["amount", "valueUsd"], aggfunc='sum'))
        else:
            print(f"Total Fees collected to date: ${total_fees:.2f}")
            print(piv_df.pivot_table(index=["tokenSymbol"], values=["amount", "valueUsd"], aggfunc='sum'))
        print(piv_df)
        return piv_df, total_fees
    
    def mark_transactions_recorded(self):
        """
        Mark all transactions as recorded
        """
        self.df.loc[:, "recorded"] = True
        # remove date column from the dataframe
        self.df = self.df.drop(columns=["date"])
        # Check to see if timeStamp exists as a column or as the index
        if "timeStamp" in self.df.columns:
            self.df.set_index("timeStamp", inplace=True)
        self.df.to_csv(self.csv_path)
    
    def analyze_lp_performance(self):
        """
        Analyze the performance of the LP
        """
        perf = {}
        for name, df in self.df_per_nft.items():
            df = df.sort_index(ascending=True)
            net_funding = df[df.transactionType == "fund"]["valueUsd"].sum() + df[df.transactionType == "dust"]["valueUsd"].sum()
            net_funding = abs(net_funding)
            total_fees = df[df.transactionType == "fee"]["valueUsd"].sum()
            # Get Hold Value and tokens
            tokens = df[df.eventType == "Deposit"]["tokenSymbol"].unique()
            for token in tokens:
                if token == "USDC":
                    continue
                else:
                    start_price = df[(df.eventType == "Deposit") & (df.tokenSymbol == token)].iloc[0].price
                    end_price = df[df.tokenSymbol == token].price.iloc[-1]
            # Check to see if LP was completed and calculate final token swaps
            tok_final = {}
            start_amount = df[(df.transactionType == "fund") & (df.tokenSymbol == token)].amount.sum()
            dust_removed = df[(df.transactionType == "dust") & (df.tokenSymbol == token)].amount.sum()
            if df['eventType'].isin(['Exit']).any():
                for token in tokens:  
                    end_amount = df[(df.transactionType == "withdraw") & (df.tokenSymbol == token)].amount.sum()
                    final = start_amount + dust_removed + end_amount
                    tok_final[token] = final
            else:
                end_amount = None
                final = None
            hold_token = net_funding / start_price
            apr = total_fees / net_funding * 365 * 100 / (df.index.max() - df.index.min()).days
            token_gain = (end_price / start_price - 1) * 100
            # Need to get the last tokenID, sorted by timeStamp
            end_token_id = int(df.tokenID.iloc[-1])
            perf[name] = {
                "$net_funding": net_funding.round(2), 
                "$total_fees": total_fees.round(2), 
                "hold_tokens": hold_token, 
                "%average_fee_apr": apr.round(1), 
                "$start_price": start_price.round(2), 
                "$final_price_csv": end_price.round(2),
                "%token_gain": token_gain.round(2),
                }
            if df['eventType'].isin(['Exit']).any():
                lp_out = {"Balances": "LP is Closed"}
                final_val = df[df.transactionType == "withdraw"]["valueUsd"].sum()
                final_price = end_price
                for key, value in tok_final.items():
                    perf[name][f"Final Balance for {key}"] = value
            else:
                lp = SickleLPTracker(end_token_id)
                lp_out = lp.balances
                final_val = lp.value
                final_price = lp.volatile_price
            perf[name]["Gain over full token hold"] = ((final_val + total_fees) - hold_token * final_price).round(2)
            perf[name]["Gain over hold USD"] = (final_val + total_fees - net_funding).round(2)
            perf[name] = {**perf[name], **lp_out}

        self.lp_analysis = pd.DataFrame(perf)
        print(self.lp_analysis)
        self.lp_analysis.to_csv(f"outputs/{self.wal_shortname}_returns.csv")
        return


class SickleLPTracker:
    """
    Simple tracker to hold the Sickle LP Information
    """
    def __init__(self, token_id):
        self.id = token_id
        info = read_sickle_contract_position(self.id)
        self.pool = info["poolAddress"]
        self.liquidity = info["liquidity"]
        self.lower_tick = info["tickLower"]
        self.upper_tick = info["tickUpper"]
        self.token0_name = info["token0Name"]
        self.token0_dec = info["token0Decimal"]
        self.token0_price = None
        self.token1_name = info["token1Name"]
        self.token1_dec = info["token1Decimal"]
        self.token1_price = None
        self.current_tick = info["tickCurrent"]
        self.value = None
        self.volatile_price = None
        self.balances = self.calc_balances()

    @property
    def price_range(self):
        lower = um.sqrtp_to_price(um.tick_to_sqrtp(self.lower_tick)) * 10 ** self.token0_dec / 10 ** self.token1_dec
        upper = um.sqrtp_to_price(um.tick_to_sqrtp(self.upper_tick)) * 10 ** self.token0_dec / 10 ** self.token1_dec
        return [lower, upper]

    def calc_balances(self):
        if self.token0_dec is None:
            raise ValueError("Token0 decimal is not set.")
        if self.token1_dec is None:
            raise ValueError("Token1 decimal is not set.")
        sq_low = um.tick_to_sqrtp(self.lower_tick)
        sq_up = um.tick_to_sqrtp(self.upper_tick)
        # token balances will come out incorrectly if the tick is not in the range
        if self.current_tick  < self.lower_tick:
            lp_sq_cur = um.tick_to_sqrtp(self.lower_tick)
        elif self.current_tick > self.upper_tick:
            lp_sq_cur = um.tick_to_sqrtp(self.upper_tick)
        else:
            lp_sq_cur = um.tick_to_sqrtp(self.current_tick)
        sq_cur = um.tick_to_sqrtp(self.current_tick)
        liq = self.liquidity
        token0_bal = um.calc_amount0(liq, lp_sq_cur, sq_up) / (10 ** self.token0_dec)
        token1_bal = um.calc_amount1(liq, lp_sq_cur, sq_low) / (10 ** self.token1_dec)
        token_dec_ratio = (10 ** self.token0_dec) / (10 ** self.token1_dec)
        lp_price = um.sqrtp_to_price(sq_cur) * token_dec_ratio
        if self.token0_name == 'USDC':
            self.token0_price = 1
            self.token1_price = 1 / lp_price
            self.volatile_price = self.token1_price
            self.value = token0_bal * self.token0_price + token1_bal * self.token1_price
        elif self.token1_name == 'USDC':
            self.token0_price = lp_price
            self.token1_price = 1
            self.volatile_price = self.token0_price
            self.value = token0_bal * self.token0_price + token1_bal * self.token1_price
        else:
            print("Neither token is USDC, can't calculate USD value.")
            self.value = 0
        balances = {
            "token0_name": self.token0_name,
            "token0_balance": token0_bal,
            "token0_price": self.token0_price,
            "token1_name": self.token1_name,
            "token1_balance": token1_bal,
            "token1_price": self.token1_price,
            "LP Value": self.value
        }
        return balances
    
    def __repr__(self):
        return (f"SickleLPTracker ID: {self.id}\n"
                f"Pool Address: {self.pool}\n"
                f"Liquidity: {self.liquidity}\n"
                f"Lower Tick: {self.lower_tick}\n"
                f"Upper Tick: {self.upper_tick}\n"
                f"Price Range: {self.price_range}\n"
                f"Current Tick: {self.current_tick}\n"
                f"Token0 Name: {self.token0_name}\n"
                f"Token0 Decimal: {self.token0_dec}\n"
                f"Token0 Price: {self.token0_price}\n"
                f"Token1 Name: {self.token1_name}\n"
                f"Token1 Decimal: {self.token1_dec}\n"
                f"Token1 Price: {self.token1_price:.2f}\n"
                f"Volatile Price: {self.volatile_price:.2f}\n"
                f"LP Value: ${self.value:.2f}\n"
                f"Balances: {self.balances}")