"""
Preprocessing and Feature Engineering for Time-Series Equity Prediction
Generates all technical indicators and features for model input
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """
    Technical indicator calculations for equity time-series data.
    """
    
    @staticmethod
    def normalize_ohlcv(
        df: pd.DataFrame,
        window: int = 200
    ) -> pd.DataFrame:
        """
        Normalize OHLCV using rolling statistics.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            window: Rolling window for normalization
            
        Returns:
            DataFrame with normalized columns
        """
        result = df.copy()
        
        # Price normalization (z-score based on close)
        close_mean = df['close'].rolling(window=window, min_periods=1).mean()
        close_std = df['close'].rolling(window=window, min_periods=1).std()
        close_std = close_std.replace(0, 1)  # Avoid division by zero
        
        for col in ['open', 'high', 'low', 'close']:
            result[f'{col}_norm'] = (df[col] - close_mean) / close_std
        
        # Volume normalization (log scale then z-score)
        log_volume = np.log1p(df['volume'])
        volume_mean = log_volume.rolling(window=window, min_periods=1).mean()
        volume_std = log_volume.rolling(window=window, min_periods=1).std()
        volume_std = volume_std.replace(0, 1)
        
        result['volume_norm'] = (log_volume - volume_mean) / volume_std
        
        return result
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False, min_periods=1).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.
        
        Args:
            series: Price series
            period: RSI period (default 14)
            
        Returns:
            RSI values (0-100)
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        period: int = 20,
        num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.
        
        Returns:
            upper, lower, middle, bandwidth
        """
        middle = series.rolling(window=period, min_periods=1).mean()
        std = series.rolling(window=period, min_periods=1).std()
        
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        bandwidth = (upper - lower) / middle.replace(0, 1e-10)
        
        return upper, lower, middle, bandwidth
    
    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence).
        
        Returns:
            macd_line, signal_line, histogram
        """
        ema_fast = TechnicalIndicators.ema(series, fast)
        ema_slow = TechnicalIndicators.ema(series, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def detect_rsi_divergence(
        price: pd.Series,
        rsi: pd.Series,
        lookback: int = 14
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Detect RSI divergences (simplified).
        
        Bullish divergence: Price makes lower low, RSI makes higher low
        Bearish divergence: Price makes higher high, RSI makes lower high
        
        Returns:
            bullish_divergence, bearish_divergence (binary signals)
        """
        bull_div = pd.Series(0, index=price.index, dtype=float)
        bear_div = pd.Series(0, index=price.index, dtype=float)
        
        for i in range(lookback, len(price)):
            # Get recent window
            price_window = price.iloc[i-lookback:i+1]
            rsi_window = rsi.iloc[i-lookback:i+1]
            
            # Find local extrema
            price_min_idx = price_window.idxmin()
            price_max_idx = price_window.idxmax()
            rsi_min_idx = rsi_window.idxmin()
            rsi_max_idx = rsi_window.idxmax()
            
            # Bullish divergence
            if price_min_idx == price_window.index[-1]:  # Current is price low
                if rsi.iloc[i] > rsi_window.min():  # RSI not at low
                    bull_div.iloc[i] = 1.0
            
            # Bearish divergence
            if price_max_idx == price_window.index[-1]:  # Current is price high
                if rsi.iloc[i] < rsi_window.max():  # RSI not at high
                    bear_div.iloc[i] = 1.0
        
        return bull_div, bear_div
    
    @staticmethod
    def support_resistance_score(
        price: pd.Series,
        window: int = 50,
        tolerance: float = 0.02
    ) -> pd.Series:
        """
        Calculate support/resistance level score (simplified).
        
        Measures how many times price has touched current level.
        
        Args:
            price: Price series
            window: Lookback window
            tolerance: Price tolerance (2% default)
            
        Returns:
            Support/resistance score
        """
        sr_score = pd.Series(0.0, index=price.index)
        
        for i in range(window, len(price)):
            current_price = price.iloc[i]
            historical_prices = price.iloc[i-window:i]
            
            # Count touches within tolerance
            lower_bound = current_price * (1 - tolerance)
            upper_bound = current_price * (1 + tolerance)
            
            touches = ((historical_prices >= lower_bound) & 
                      (historical_prices <= upper_bound)).sum()
            
            # Normalize score
            sr_score.iloc[i] = touches / window
        
        return sr_score
    
    @staticmethod
    def supply_demand_zone_score(
        df: pd.DataFrame,
        window: int = 20,
        volume_threshold: float = 1.5
    ) -> pd.Series:
        """
        Calculate supply/demand zone score (simplified).
        
        High volume + price rejection = strong zone
        
        Args:
            df: DataFrame with OHLCV data
            window: Lookback window
            volume_threshold: Volume multiplier for significance
            
        Returns:
            Supply/demand zone score
        """
        sd_score = pd.Series(0.0, index=df.index)
        
        # Volume spike detection
        avg_volume = df['volume'].rolling(window=window, min_periods=1).mean()
        volume_spike = df['volume'] > (avg_volume * volume_threshold)
        
        # Price rejection (wick size relative to body)
        body_size = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        wick_ratio = (total_range - body_size) / total_range.replace(0, 1e-10)
        
        # Score combines volume and rejection
        sd_score = (volume_spike.astype(float) * wick_ratio).fillna(0)
        
        return sd_score


class FeatureEngineering:
    """
    Complete feature engineering pipeline for time-series model.
    """
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.feature_names = []
    
    def create_features(
        self,
        df: pd.DataFrame,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Create streamlined features from OHLCV data (18 features total).

        Reduced from 32 to 18 features to minimize noise and redundancy.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            normalize: Whether to normalize features

        Returns:
            DataFrame with 18 engineered features
        """
        result = pd.DataFrame(index=df.index)

        # 1. Core OHLCV (5 features)
        if normalize:
            norm_df = self.indicators.normalize_ohlcv(df, window=100)
            result['open_norm'] = norm_df['open_norm']
            result['high_norm'] = norm_df['high_norm']
            result['low_norm'] = norm_df['low_norm']
            result['close_norm'] = norm_df['close_norm']
            result['volume_norm'] = norm_df['volume_norm']
        else:
            result['open'] = df['open']
            result['high'] = df['high']
            result['low'] = df['low']
            result['close'] = df['close']
            result['volume'] = df['volume']

        # 2. Moving Averages (2 features)
        sma_100 = self.indicators.sma(df['close'], 100)
        sma_50 = self.indicators.sma(df['close'], 50)

        result['sma_100_ratio'] = (sma_100 / df['close']) - 1
        result['sma_50_ratio'] = (sma_50 / df['close']) - 1

        # 3. RSI (1 feature - removed divergences as too noisy)
        rsi = self.indicators.rsi(df['close'], period=14)
        result['rsi'] = (rsi - 50) / 50  # Normalize to [-1, 1]

        # 4. Bollinger Bands (2 features - simplified)
        bb_upper, bb_lower, bb_middle, bb_width = self.indicators.bollinger_bands(
            df['close'], period=20, num_std=2.0
        )

        # Position within bands [-1, 1]
        result['bb_position'] = ((df['close'] - bb_lower) /
                                 (bb_upper - bb_lower).replace(0, 1) * 2) - 1
        result['bb_width'] = bb_width

        # 5. MACD (1 feature - just histogram)
        macd_line, signal_line, histogram = self.indicators.macd(df['close'])
        result['macd_histogram'] = histogram / df['close']

        # 6. Price Momentum (2 features - keep short and medium term)
        result['return_1d'] = df['close'].pct_change(1)
        result['return_5d'] = df['close'].pct_change(5)

        # 7. Volatility (1 feature)
        result['volatility_20d'] = df['close'].pct_change().rolling(20).std()

        # 8. Volume (1 feature)
        result['volume_ma_ratio'] = (df['volume'] /
                                     df['volume'].rolling(20, min_periods=1).mean()) - 1

        # 9. Price Patterns (2 features)
        result['high_low_range'] = (df['high'] - df['low']) / df['close']
        result['close_open_change'] = (df['close'] - df['open']) / df['open']

        # 10. Trend Strength (1 feature)
        ema_12 = self.indicators.ema(df['close'], 12)
        ema_26 = self.indicators.ema(df['close'], 26)
        result['ema_diff'] = (ema_12 - ema_26) / df['close']

        # Fill NaN values
        result = result.fillna(method='ffill').fillna(0)

        # Store feature names
        self.feature_names = result.columns.tolist()

        # Validate we have exactly 18 features
        assert len(self.feature_names) == 18, f"Expected 18 features, got {len(self.feature_names)}"

        return result
    
    def create_targets(self, df: pd.DataFrame, clip_range: tuple = (-0.5, 0.5)) -> pd.Series:
        """
        Create target variable: next candle return (clipped to remove outliers).

        Args:
            df: DataFrame with 'close' column
            clip_range: (min, max) tuple for clipping targets (default: ±50%)

        Returns:
            Series with next candle returns (clipped)
        """
        # Next candle return: (close[t+1] - close[t]) / close[t]
        targets = df['close'].pct_change(1).shift(-1)

        # Clip extreme outliers (stock splits, errors, extreme events)
        # Default: clip to ±50% returns to remove data quality issues
        targets = targets.clip(lower=clip_range[0], upper=clip_range[1])

        return targets
    
    def get_feature_count(self) -> int:
        """Get total number of features."""
        return len(self.feature_names)


class TimeSeriesDataset:
    """
    Dataset class for creating sequences for model training.
    """
    
    def __init__(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        seq_len: int = 200
    ):
        """
        Args:
            features: DataFrame with all features
            targets: Series with target values
            seq_len: Sequence length for model input
        """
        self.features = features.values
        self.targets = targets.values
        self.seq_len = seq_len
        
        # Calculate valid indices (must have full sequence)
        self.valid_indices = list(range(seq_len, len(features)))
    
    def __len__(self) -> int:
        """Number of valid sequences."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, float]:
        """
        Get a sequence and its target.
        
        Returns:
            features: (seq_len, n_features)
            target: scalar
        """
        actual_idx = self.valid_indices[idx]
        
        # Get sequence
        sequence = self.features[actual_idx - self.seq_len:actual_idx]
        
        # Get target
        target = self.targets[actual_idx]
        
        return sequence, target
    
    def get_batch(
        self,
        batch_indices: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a batch of sequences.
        
        Returns:
            features: (batch_size, seq_len, n_features)
            targets: (batch_size, 1)
        """
        sequences = []
        targets = []
        
        for idx in batch_indices:
            seq, target = self.__getitem__(idx)
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets).reshape(-1, 1)


def create_sample_data(
    n_samples: int = 5000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Create synthetic OHLCV data for testing.
    
    Args:
        n_samples: Number of time steps
        seed: Random seed
        
    Returns:
        DataFrame with OHLCV columns
    """
    np.random.seed(seed)
    
    # Generate synthetic price data with trend and noise
    base_price = 100.0
    trend = np.linspace(0, 20, n_samples)
    noise = np.cumsum(np.random.randn(n_samples) * 0.5)
    
    close = base_price + trend + noise
    
    # Generate OHLC from close
    volatility = 0.02
    high = close * (1 + np.abs(np.random.randn(n_samples)) * volatility)
    low = close * (1 - np.abs(np.random.randn(n_samples)) * volatility)
    
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    
    # Generate volume
    base_volume = 1_000_000
    volume = base_volume * (1 + np.random.randn(n_samples) * 0.3)
    volume = np.abs(volume)
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return df


if __name__ == "__main__":
    print("=" * 80)
    print("Feature Engineering Pipeline Test")
    print("=" * 80)
    
    # Create sample data
    print("\n1. Creating synthetic OHLCV data...")
    df = create_sample_data(n_samples=5000, seed=42)
    print(f"   Data shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"\n   Sample data:")
    print(df.head())
    
    # Initialize feature engineering
    print("\n2. Engineering features...")
    fe = FeatureEngineering()
    features = fe.create_features(df, normalize=True)
    
    print(f"   Feature shape: {features.shape}")
    print(f"   Number of features: {fe.get_feature_count()}")
    print(f"\n   Feature names:")
    for i, name in enumerate(fe.feature_names, 1):
        print(f"      {i:2d}. {name}")
    
    # Create targets
    print("\n3. Creating targets...")
    targets = fe.create_targets(df)
    print(f"   Target shape: {targets.shape}")
    print(f"   Target statistics:")
    print(f"      Mean: {targets.mean():.6f}")
    print(f"      Std:  {targets.std():.6f}")
    print(f"      Min:  {targets.min():.6f}")
    print(f"      Max:  {targets.max():.6f}")
    
    # Create dataset
    print("\n4. Creating time-series dataset...")
    dataset = TimeSeriesDataset(features, targets, seq_len=200)
    print(f"   Total sequences: {len(dataset)}")
    
    # Get sample batch
    batch_size = 8
    batch_indices = list(range(batch_size))
    X_batch, y_batch = dataset.get_batch(batch_indices)
    
    print(f"\n5. Sample batch:")
    print(f"   Features shape: {X_batch.shape}  # (batch, seq_len, n_features)")
    print(f"   Targets shape:  {y_batch.shape}  # (batch, 1)")
    
    print(f"\n   Feature statistics (first sequence):")
    print(f"      Mean: {X_batch[0].mean():.6f}")
    print(f"      Std:  {X_batch[0].std():.6f}")
    print(f"      Min:  {X_batch[0].min():.6f}")
    print(f"      Max:  {X_batch[0].max():.6f}")
    
    print(f"\n   Sample targets: {y_batch.squeeze()[:5]}")
    
    # Check for NaN or Inf
    has_nan = np.isnan(X_batch).any() or np.isnan(y_batch).any()
    has_inf = np.isinf(X_batch).any() or np.isinf(y_batch).any()
    
    print(f"\n6. Data quality check:")
    print(f"   Contains NaN: {has_nan}")
    print(f"   Contains Inf: {has_inf}")
    
    if not has_nan and not has_inf:
        print("   ✓ Data is clean and ready for training!")
    
    print("\n" + "=" * 80)
    print("Feature engineering test completed successfully!")
    print("=" * 80)