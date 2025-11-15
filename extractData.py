"""
Data Extraction Module
Loads raw ticker data from text files and converts to DataFrame format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class TickerDataExtractor:
    """
    Extracts and loads ticker data from raw text files.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: Directory containing ticker .txt files
        """
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

    def load_ticker(
        self,
        ticker: str,
        min_samples: int = 1000
    ) -> Optional[pd.DataFrame]:
        """
        Load data for a single ticker.

        Args:
            ticker: Ticker symbol (e.g., 'AAPL')
            min_samples: Minimum number of samples required

        Returns:
            DataFrame with OHLCV data, or None if insufficient data
        """
        file_path = self.data_dir / f"{ticker}.txt"

        if not file_path.exists():
            print(f"Warning: File not found for {ticker}")
            return None

        try:
            # Read CSV with appropriate column names
            df = pd.read_csv(
                file_path,
                names=['date', 'open', 'high', 'low', 'close', 'volume'],
                parse_dates=['date'],
                index_col='date'
            )

            # Data validation
            if len(df) < min_samples:
                print(f"Warning: {ticker} has only {len(df)} samples (minimum: {min_samples})")
                return None

            # Check for required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: {ticker} missing required columns")
                return None

            # Remove rows with NaN or invalid values
            df = df.dropna()
            df = df[(df[required_cols] > 0).all(axis=1)]  # All prices/volume must be positive

            # Ensure OHLC relationship is valid
            df = df[
                (df['high'] >= df['low']) &
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close'])
            ]

            # Sort by date (ascending - time series order)
            df = df.sort_index()

            # Final check
            if len(df) < min_samples:
                print(f"Warning: {ticker} has only {len(df)} valid samples after cleaning")
                return None

            return df

        except Exception as e:
            print(f"Error loading {ticker}: {e}")
            return None

    def get_available_tickers(self, max_length: int = 4) -> list:
        """
        Get list of available tickers from data directory.

        Args:
            max_length: Maximum ticker length (default: 4, for standard tickers)

        Returns:
            List of ticker symbols
        """
        tickers = []

        for file_path in self.data_dir.glob("*.txt"):
            ticker = file_path.stem

            # Filter: only alphabetic, 1-4 characters
            if ticker.isalpha() and 1 <= len(ticker) <= max_length:
                tickers.append(ticker)

        return sorted(tickers)

    def get_ticker_info(self, ticker: str) -> Dict:
        """
        Get information about a ticker's data.

        Args:
            ticker: Ticker symbol

        Returns:
            Dictionary with ticker information
        """
        df = self.load_ticker(ticker, min_samples=1)

        if df is None:
            return {
                'ticker': ticker,
                'available': False
            }

        return {
            'ticker': ticker,
            'available': True,
            'samples': len(df),
            'start_date': df.index[0].strftime('%Y-%m-%d'),
            'end_date': df.index[-1].strftime('%Y-%m-%d'),
            'date_range_days': (df.index[-1] - df.index[0]).days,
            'avg_price': df['close'].mean(),
            'avg_volume': df['volume'].mean()
        }


if __name__ == "__main__":
    print("=" * 80)
    print("Ticker Data Extraction Test")
    print("=" * 80)

    # Initialize extractor
    extractor = TickerDataExtractor(data_dir="data")

    # Get available tickers
    print("\n1. Scanning for available tickers...")
    tickers = extractor.get_available_tickers(max_length=4)
    print(f"   Found {len(tickers)} valid tickers (1-4 letters, alphabetic)")

    if len(tickers) > 0:
        print(f"\n   First 20 tickers: {tickers[:20]}")

    # Test loading a specific ticker
    if 'AAPL' in tickers:
        print("\n2. Loading AAPL data...")
        df = extractor.load_ticker('AAPL', min_samples=1000)

        if df is not None:
            print(f"   Shape: {df.shape}")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            print(f"   Total days: {(df.index[-1] - df.index[0]).days}")
            print(f"\n   First 5 rows:")
            print(df.head())
            print(f"\n   Data statistics:")
            print(df.describe())

            # Get ticker info
            info = extractor.get_ticker_info('AAPL')
            print(f"\n   Ticker info:")
            for key, value in info.items():
                print(f"      {key}: {value}")

    # Test multiple tickers
    if len(tickers) >= 5:
        print(f"\n3. Testing batch load (first 5 tickers)...")
        for ticker in tickers[:5]:
            df = extractor.load_ticker(ticker, min_samples=1000)
            if df is not None:
                print(f"   ✓ {ticker:6s}: {len(df):5d} samples, "
                      f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            else:
                print(f"   ✗ {ticker:6s}: Insufficient data")

    print("\n" + "=" * 80)
    print("Data extraction test completed!")
    print("=" * 80)
