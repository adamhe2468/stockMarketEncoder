# Stock Data Processing Pipeline

Complete pipeline for extracting, preprocessing, and loading stock data for time-series prediction.

## Pipeline Overview

```
Raw Data (.txt files)
    ↓
[1. extractData.py] - Load raw ticker data
    ↓
[2. preprocessAllTickers.py] - Create features & sequences
    ↓
Processed Data (.npz files per ticker)
    ↓
[3. dataLoader.py] - Load batches for training
    ↓
Model Training
```

## Files Created

1. **extractData.py** - Loads raw ticker data from .txt files
2. **preprocessAllTickers.py** - Processes all tickers and creates NPZ files
3. **dataLoader.py** - PyTorch DataLoader for training
4. **filterTickers.py** - Filter ticker lists (utility)
5. **run_filter_tickers.py** - Filter data folder files (utility)

## Quick Start

### Step 1: Filter Tickers (Optional)

Remove invalid tickers (>4 letters or special characters):

```bash
python run_filter_tickers.py
```

### Step 2: Preprocess All Tickers

Process all tickers and create NPZ files:

```bash
python preprocessAllTickers.py
```

This will:
- Load each ticker from `data/` folder
- Create technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Generate time-aligned sequences (length=200)
- Split chronologically: 70% train, 15% val, 15% test
- Save to `processed_data/{TICKER}.npz`

### Step 3: Load Data for Training

```python
from dataLoader import MultiTickerDataLoader

# Initialize loader
loader = MultiTickerDataLoader(
    processed_data_dir="processed_data",
    batch_size=32,
    num_workers=0
)

# Create DataLoaders (shuffle=False for time-series!)
train_loader, val_loader, test_loader = loader.get_dataloaders()

# Use in training loop
for sequences, targets in train_loader:
    # sequences: (batch, seq_len=200, n_features=18)
    # targets: (batch, 1)
    predictions = model(sequences)
    loss = criterion(predictions, targets)
    # ...
```

## Data Format

### Raw Data (.txt files)
```
date,open,high,low,close,volume
05/01/2000,1.115,1.1172,1.08817,1.1099,216148800
...
```

### Processed Data (.npz files)
Each ticker's NPZ file contains:
- `X_train`: (n_train, 200, 18) - Training sequences
- `y_train`: (n_train, 1) - Training targets
- `timestamps_train`: Training timestamps
- `X_val`: Validation sequences
- `y_val`: Validation targets
- `timestamps_val`: Validation timestamps
- `X_test`: Test sequences
- `y_test`: Test targets
- `timestamps_test`: Test timestamps
- Metadata: ticker, seq_len, n_features, feature_names

## Features (18 total)

**Streamlined feature set - reduced from 32 to minimize noise and redundancy**

1. **Normalized OHLCV** (5)
   - open_norm, high_norm, low_norm, close_norm, volume_norm

2. **Moving Averages** (2)
   - sma_100_ratio, sma_50_ratio

3. **RSI** (1)
   - rsi (normalized to [-1, 1])

4. **Bollinger Bands** (2)
   - bb_position, bb_width

5. **MACD** (1)
   - macd_histogram

6. **Momentum** (2)
   - return_1d, return_5d

7. **Volatility** (1)
   - volatility_20d

8. **Volume** (1)
   - volume_ma_ratio

9. **Price Patterns** (2)
   - high_low_range, close_open_change

10. **Trend Strength** (1)
    - ema_diff

## Target Variable

- **Next candle return**: `(close[t+1] - close[t]) / close[t]`

## Time-Series Considerations

### CRITICAL: NO SHUFFLING

Time-series data maintains temporal order:

✅ **Correct**:
```python
train_loader = DataLoader(dataset, shuffle=False)  # ✓
```

❌ **Wrong**:
```python
train_loader = DataLoader(dataset, shuffle=True)  # ✗ Destroys temporal structure!
```

### Chronological Split

Data is split by time, not randomly:

```
|────────── Train (70%) ──────────|── Val (15%) ──|── Test (15%) ──|
├─────────────────────────────────────────────────────────────────►
2000                           2017     2020   2023    Time
```

### Sequence Creation

Sequences are created in temporal order:

```
Sample 1: [day 0:200] → target at day 200
Sample 2: [day 1:201] → target at day 201
Sample 3: [day 2:202] → target at day 202
...
```

## Configuration

### preprocessAllTickers.py

```python
preprocessor = TickerPreprocessor(
    data_dir="data",              # Raw data folder
    output_dir="processed_data",  # Output folder
    seq_len=200,                  # Sequence length
    min_samples=1500              # Minimum samples required
)

results = preprocessor.process_all_tickers(
    max_tickers=None,   # None = all tickers
    train_ratio=0.70,   # 70% training
    val_ratio=0.15      # 15% validation, 15% test
)
```

### dataLoader.py

```python
loader = MultiTickerDataLoader(
    processed_data_dir="processed_data",
    batch_size=32,       # Batch size
    num_workers=0        # DataLoader workers
)
```

## Example: Full Pipeline

```python
# 1. Preprocess all tickers
from preprocessAllTickers import TickerPreprocessor

preprocessor = TickerPreprocessor()
results = preprocessor.process_all_tickers()

# 2. Load data for training
from dataLoader import MultiTickerDataLoader

loader = MultiTickerDataLoader(batch_size=32)
train_loader, val_loader, test_loader = loader.get_dataloaders()

# 3. Get dataset info
info = loader.get_dataset_info()
print(f"Tickers: {info['n_tickers']}")
print(f"Features: {info['n_features']}")
print(f"Sequence length: {info['seq_len']}")

# 4. Train model
for epoch in range(epochs):
    for sequences, targets in train_loader:
        # sequences: (batch, 200, 18)
        # targets: (batch, 1)

        predictions = model(sequences)
        loss = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Troubleshooting

### No NPZ files found
```
Error: No NPZ files found in processed_data
```
→ Run `preprocessAllTickers.py` first

### Insufficient data
```
Warning: TICKER has only 500 samples (minimum: 1500)
```
→ Ticker has insufficient historical data, will be skipped

### NaN in features
```
Contains NaN: True
```
→ Check preprocessing, NaN values should be filled automatically

## Performance

- **Preprocessing**: ~1-2 seconds per ticker
- **Loading**: Instant (data precomputed)
- **Memory**: ~10MB per ticker NPZ file
- **Disk**: ~2GB for 200 tickers

## Notes

- All features are normalized/standardized
- No look-ahead bias (features only use past data)
- Targets are forward-looking (next candle return)
- Time alignment is strictly maintained
- **Never shuffle time-series data!**
