import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

TOKEN = 'btc'

# Load data
def load_data():
    df = pd.read_csv(f'data/binance_{TOKEN}_minute.csv', parse_dates=['date'], index_col='date')
    return df

# Function to find local volatility peaks within a given quantile range
def find_volatility_peaks(df, quantile_low, quantile_high, order=100):
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df['volatility'] = df['log_return'].rolling('12h').std()
    df['smoothed_volatility'] = df['volatility'].rolling('24h').mean()
    
    vol_min = df['smoothed_volatility'].quantile(quantile_low)
    vol_max = df['smoothed_volatility'].quantile(quantile_high)
    df_filtered = df[(df['smoothed_volatility'] >= vol_min) & (df['smoothed_volatility'] <= vol_max)]
    
    peaks = argrelextrema(df_filtered['smoothed_volatility'].values, np.greater, order=order)[0]
    # Order the peaks in terms of the highest volatility they represent in the dataframe
    if quantile_high  > 0.5:
        peaks = peaks[-np.argsort(df_filtered['smoothed_volatility'].iloc[peaks])]
    else:
        peaks = np.random.choice(peaks, size=len(peaks), replace=False)

    # Randomize the order of the peaks
    # peaks = np.random.choice(peaks, size=len(peaks), replace=False)

    return df_filtered.iloc[peaks]

# Extract a single representative period per category
def extract_period(df, peak_df, duration, trend):
    for peak_time in peak_df.index:
        start_time = peak_time - pd.Timedelta(days=duration / 2)
        end_time = peak_time + pd.Timedelta(days=duration / 2)
        period = df.loc[start_time:end_time]
        if (period['price'].iloc[-1] > period['price'].iloc[0]) == trend:
            return period
    return None

# Plot extracted periods
def plot_extracted_periods(periods):
    plt.figure(figsize=(12, 6))
    for label, period in periods.items():
        plt.plot(period.index, period['price'], label=f'{label}')
    plt.legend()
    plt.grid()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Extracted Periods')
    plt.show()

# Main function
def main():
    df = load_data()
    
    periods = {
        'high_vol_up': extract_period(df, find_volatility_peaks(df, 0.98, 1.0), 14, True),
        'high_vol_down': extract_period(df, find_volatility_peaks(df, 0.98, 1.0), 14, False),
        'med_vol_up': extract_period(df, find_volatility_peaks(df, 0.4, 0.66), 14, True),
        'med_vol_down': extract_period(df, find_volatility_peaks(df, 0.4, 0.66), 14, False),
        'low_vol_up': extract_period(df, find_volatility_peaks(df, 0.0, 0.33), 14, True),
        'low_vol_down': extract_period(df, find_volatility_peaks(df, 0.0, 0.33), 14, False)
    }
    
    for label, period in periods.items():
        if period is not None:
            print(f'Extracted {label} period: {period.index[0]} : {period.index[-1]}')
            period.to_csv(f'data/volatility_segments/{TOKEN}_{label}.csv')
    
    plot_extracted_periods(periods)
    
    print("Extracted periods saved to CSV files and plotted.")
    
if __name__ == "__main__":
    main()
