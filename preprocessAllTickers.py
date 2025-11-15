"""
Preprocessing Pipeline for All Tickers
Processes each ticker's data and saves as NPZ files for efficient training.

CRITICAL: Time-series data - maintains temporal order, NO SHUFFLING.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from extractData import TickerDataExtractor
from initPreprocess import FeatureEngineering


class TickerPreprocessor:
    """
    Preprocesses individual ticker data and saves to NPZ format.
    """

    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "processed_data",
        seq_len: int = 200,
        min_samples: int = 0  # 200 seq_len + buffer
    ):
        """
        Args:
            data_dir: Directory with raw ticker .txt files
            output_dir: Directory to save processed .npz files
            seq_len: Sequence length for model input
            min_samples: Minimum samples required per ticker
        """
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.seq_len = seq_len
        self.min_samples = min_samples

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.extractor = TickerDataExtractor(data_dir)
        self.feature_eng = FeatureEngineering()

    def preprocess_ticker(
        self,
        ticker: str,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15
    ) -> Dict:
        """
        Preprocess a single ticker and save to NPZ file.

        IMPORTANT: Time-series split - chronological order maintained!

        Args:
            ticker: Ticker symbol
            train_ratio: Training set ratio (0.70 = 70%)
            val_ratio: Validation set ratio (0.15 = 15%)

        Returns:
            Dictionary with processing statistics
        """
        # Load raw data
        df = self.extractor.load_ticker(ticker, min_samples=self.min_samples)

        if df is None:
            return {
                'ticker': ticker,
                'status': 'failed',
                'reason': 'insufficient_data'
            }

        try:
            # Create features
            features = self.feature_eng.create_features(df, normalize=True)

            # Create targets with clipping (±50% to remove extreme outliers)
            targets_raw = self.feature_eng.create_targets(df, clip_range=(-0.5, 0.5))

            # Track clipping statistics
            n_clipped = ((targets_raw == 0.5) | (targets_raw == -0.5)).sum()

            # Remove last sample (no target available)
            features = features.iloc[:-1]
            targets = targets_raw.iloc[:-1]

            # Remove NaN targets
            valid_mask = ~targets.isna()
            features = features[valid_mask]
            targets = targets[valid_mask]

            # Ensure we have enough data after preprocessing
            if len(features) < self.min_samples:
                return {
                    'ticker': ticker,
                    'status': 'failed',
                    'reason': 'insufficient_after_preprocessing',
                    'clipped_count': int(n_clipped)
                }

            # Create sequences (time-aligned)
            X_sequences, y_targets, timestamps = self._create_sequences(
                features, targets
            )

            if len(X_sequences) == 0:
                return {
                    'ticker': ticker,
                    'status': 'failed',
                    'reason': 'no_valid_sequences'
                }

            # Time-series split (chronological - NO SHUFFLING!)
            train_size = int(len(X_sequences) * train_ratio)
            val_size = int(len(X_sequences) * val_ratio)
            test_size = len(X_sequences) - train_size - val_size

            # Split data chronologically
            X_train = X_sequences[:train_size]
            y_train = y_targets[:train_size]
            timestamps_train = timestamps[:train_size]

            X_val = X_sequences[train_size:train_size + val_size]
            y_val = y_targets[train_size:train_size + val_size]
            timestamps_val = timestamps[train_size:train_size + val_size]

            X_test = X_sequences[train_size + val_size:]
            y_test = y_targets[train_size + val_size:]
            timestamps_test = timestamps[train_size + val_size:]

            # Save to NPZ file
            output_path = self.output_dir / f"{ticker}.npz"
            np.savez_compressed(
                output_path,
                # Training data
                X_train=X_train.astype(np.float32),
                y_train=y_train.astype(np.float32),
                timestamps_train=timestamps_train,
                # Validation data
                X_val=X_val.astype(np.float32),
                y_val=y_val.astype(np.float32),
                timestamps_val=timestamps_val,
                # Test data
                X_test=X_test.astype(np.float32),
                y_test=y_test.astype(np.float32),
                timestamps_test=timestamps_test,
                # Metadata
                ticker=ticker,
                seq_len=self.seq_len,
                n_features=X_train.shape[2],
                feature_names=self.feature_eng.feature_names,
                train_ratio=train_ratio,
                val_ratio=val_ratio
            )

            return {
                'ticker': ticker,
                'status': 'success',
                'total_sequences': len(X_sequences),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'n_features': X_train.shape[2],
                'seq_len': self.seq_len,
                'clipped_count': int(n_clipped),
                'clipped_pct': float(n_clipped / len(targets_raw) * 100) if len(targets_raw) > 0 else 0.0,
                'output_file': str(output_path),
                'date_range': f"{timestamps[0]} to {timestamps[-1]}"
            }

        except Exception as e:
            return {
                'ticker': ticker,
                'status': 'failed',
                'reason': f'error: {str(e)}'
            }

    def _create_sequences(
        self,
        features: pd.DataFrame,
        targets: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create time-aligned sequences from features and targets.

        CRITICAL: Maintains temporal order - sequences are created chronologically.

        Args:
            features: DataFrame with features
            targets: Series with targets

        Returns:
            X: (n_sequences, seq_len, n_features)
            y: (n_sequences, 1)
            timestamps: (n_sequences,) - end timestamp of each sequence
        """
        features_array = features.values
        targets_array = targets.values
        timestamps_array = features.index.values

        n_samples = len(features_array)
        n_sequences = n_samples - self.seq_len

        if n_sequences <= 0:
            return np.array([]), np.array([]), np.array([])

        # Preallocate arrays
        X = np.zeros((n_sequences, self.seq_len, features_array.shape[1]), dtype=np.float32)
        y = np.zeros((n_sequences, 1), dtype=np.float32)
        timestamps = np.zeros(n_sequences, dtype='datetime64[ns]')

        # Create sequences in temporal order
        for i in range(n_sequences):
            # Sequence: [i : i + seq_len]
            X[i] = features_array[i:i + self.seq_len]

            # Target: at position i + seq_len
            y[i, 0] = targets_array[i + self.seq_len]

            # Timestamp: end of sequence
            timestamps[i] = timestamps_array[i + self.seq_len]

        return X, y, timestamps

    def process_all_tickers(
        self,
        max_tickers: int = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15
    ) -> pd.DataFrame:
        """
        Process all available tickers and save to NPZ files.

        Args:
            max_tickers: Maximum number of tickers to process (None = all)
            train_ratio: Training set ratio
            val_ratio: Validation set ratio

        Returns:
            DataFrame with processing statistics for all tickers
        """
        # Get available tickers
        tickers = self.extractor.get_available_tickers(max_length=4)

        if max_tickers is not None:
            tickers = tickers[:max_tickers]

        print(f"Processing {len(tickers)} tickers...")
        print("=" * 80)

        results = []

        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")

            result = self.preprocess_ticker(
                ticker,
                train_ratio=train_ratio,
                val_ratio=val_ratio
            )

            results.append(result)

            # Print status
            if result['status'] == 'success':
                print(f"  ✓ Success: {result['train_samples']} train, "
                      f"{result['val_samples']} val, {result['test_samples']} test samples")
            else:
                print(f"  ✗ Failed: {result.get('reason', 'unknown')}")

        # Create summary DataFrame
        df_results = pd.DataFrame(results)

        # Print summary
        print("\n" + "=" * 80)
        print("PROCESSING SUMMARY")
        print("=" * 80)

        success_count = (df_results['status'] == 'success').sum()
        print(f"Total tickers processed: {len(tickers)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(tickers) - success_count}")

        if success_count > 0:
            success_df = df_results[df_results['status'] == 'success']
            print(f"\nTotal samples:")
            print(f"  Training:   {success_df['train_samples'].sum():,}")
            print(f"  Validation: {success_df['val_samples'].sum():,}")
            print(f"  Test:       {success_df['test_samples'].sum():,}")
            print(f"  Grand Total: {success_df['total_sequences'].sum():,}")

        # Save summary
        summary_path = self.output_dir / "preprocessing_summary.csv"
        df_results.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")

        return df_results


if __name__ == "__main__":
    print("=" * 80)
    print("Ticker Preprocessing Pipeline")
    print("=" * 80)
    print("\nConfiguration:")
    print("  - Sequence length: 200")
    print("  - Train/Val/Test split: 70/15/15 (chronological)")
    print("  - NO SHUFFLING (time-series data)")
    print("  - Time-aligned sequences")
    print("=" * 80)
    try:
    # Initialize preprocessor
        preprocessor = TickerPreprocessor(
            data_dir="data",
            output_dir="processed_data2real",
            seq_len=200,
            min_samples=0
        )

        # Process all tickers (or specify max_tickers for testing)
        # For testing, use max_tickers=10
        # For full processing, use max_tickers=None
        results = preprocessor.process_all_tickers(
            max_tickers=None,  # Process ALL tickers
            train_ratio=0.70,
            val_ratio=0.15
        )
    except Exception as e:
        print(f"Error during preprocessing: {e}")

    print("\n" + "=" * 80)
    print("Preprocessing completed!")
    print("=" * 80)
