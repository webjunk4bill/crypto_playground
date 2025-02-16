import numpy as np
import pandas as pd
import datetime
import yfinance as yf

def load_historic_data(symbol, start_date_str, today_date_str, period, interval, prepost):
    try:
        df = yf.download(symbol, start=start_date_str, end=today_date_str, period=period, interval=interval, prepost=prepost)
        #  Add symbol
        df["Symbol"] = symbol
        return df
    except:
        print('Error loading stock data for ' + symbol)
        return None



def load():
    # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    # fetch data by interval (including intraday if period < 60 days)
            # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    period = '28d'
    interval = '1m'
    prepost = True
    today = datetime.date.today()
    today_date_str = today.strftime("%Y-%m-%d")
    #  NOTE: 7 days is the max allowed
    start_days = datetime.timedelta(28)
    start_date = today - start_days
    start_date_str = datetime.datetime.strftime(start_date, "%Y-%m-%d")

    end_days = datetime.timedelta(21)
    end_date = today - end_days
    end_date_str = datetime.datetime.strftime(end_date, "%Y-%m-%d")

    #  Coins to download
    symbol = 'BTC-USD'

    #  Fetch data for coin symbols

    print(f"Loading data for {symbol}")
    df = load_historic_data(symbol, start_date_str, end_date_str, period, interval, prepost)
    df = cleanup(df)
    file_name = f"data/{today_date_str}_{symbol}_{period}_{interval}.csv"
    df.to_csv(f"{file_name}")


def cleanup(df):
    df.columns = df.columns.droplevel(1)
    df = df.reset_index().rename(columns={'Datetime': 'Date', 'Open': 'Price'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').asfreq('1min')
    df['Volume'] = df['Volume'].replace(0, np.nan)
    df['Volume'] = df['Volume'].interpolate().bfill()
    df.bfill(inplace=True)
    return df[['Price', 'Volume']]

if __name__ == '__main__':
    load()

