import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load data
def load_data():
    df = pd.read_csv('data/binance_btc_minute.csv', parse_dates=['date'], index_col='date')
    return df

def segment_by_volatility(df):
    # Calculate log returns
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))

    # Compute 12-hour rolling volatility
    df['volatility'] = df['log_return'].rolling('12h').std()

    # Smooth volatility using a 24-hour rolling average
    df['smoothed_volatility'] = df['volatility'].rolling('24h').mean()

    # Define volatility thresholds (heavily weighted towards lower volatility)
    low_threshold = df['smoothed_volatility'].quantile(0.66)
    high_threshold = df['smoothed_volatility'].quantile(0.95)

    # Identify local peaks in high volatility
    peaks, _ = find_peaks(df['smoothed_volatility'], height=high_threshold, distance=3*24*60)
    df['segment'] = np.nan

    # Assign high volatility segments (3-day windows around peaks)
    for peak in peaks:
        start = max(0, peak - (1.5 * 24 * 60))
        end = min(len(df) - 1, peak + (1.5 * 24 * 60))
        df.iloc[int(start):int(end), df.columns.get_loc('segment')] = 3

    # Extract remaining unassigned data for medium volatility segmentation
    remaining_df = df[df['segment'].isna()].copy()
    peaks, _ = find_peaks(remaining_df['smoothed_volatility'], height=low_threshold, distance=7*24*60)

    # Assign medium volatility segments (7-day windows around peaks)
    for peak in peaks:
        start = max(0, peak - (3.5 * 24 * 60))
        end = min(len(remaining_df) - 1, peak + (3.5 * 24 * 60))
        remaining_df.iloc[int(start):int(end), remaining_df.columns.get_loc('segment')] = 7

    # Merge back medium volatility assignments
    df.update(remaining_df)

    # Assign remaining low volatility segments and break into 30-day segments
    df.loc[df['segment'].isna(), 'segment'] = 30
    df['final_segment'] = (df['segment'] != df['segment'].shift()).cumsum()

    # Group by segment and analyze
    segments = [group for _, group in df.groupby('final_segment')]

    return df, segments

# Main function
def main():
    df = load_data()
    df, segments = segment_by_volatility(df)
    df.to_csv('data/segmented_volatility.csv')
    
    for i, segment in enumerate(segments[:10]):  # Show first 10 segments
        print(f"Segment {i + 1}: {segment.index[0]} to {segment.index[-1]} ({len(segment)} rows)")
    
    # Plot segments over time
    plt.figure(figsize=(12, 5))
    plt.scatter(df.index, df['smoothed_volatility'], c=df['segment'], cmap='viridis', alpha=0.5)
    plt.title('Segmented Volatility Over Time')
    plt.xlabel('Date')
    plt.ylabel('Smoothed Volatility')
    plt.colorbar(label='Segment Type')
    plt.show()
    
if __name__ == "__main__":
    main()
