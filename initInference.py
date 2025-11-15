"""
Inference Script for Time-Series Equity Prediction Model
Load trained model and make predictions on new data
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Tuple
from pathlib import Path

from modlPred import TimeSeriesMambaModel
from initPreprocess import FeatureEngineering


class TimeSeriesPredictor:
    """
    Predictor class for inference on new time-series data.
    """
    
    def __init__(
        self,
        model: TimeSeriesMambaModel,
        feature_engineer: FeatureEngineering,
        device: torch.device,
        seq_len: int = 200
    ):
        """
        Args:
            model: Trained model
            feature_engineer: Feature engineering pipeline
            device: Device to run inference on
            seq_len: Sequence length for predictions
        """
        self.model = model.to(device)
        self.model.eval()
        self.fe = feature_engineer
        self.device = device
        self.seq_len = seq_len
    
    def predict_next_return(
        self,
        df: pd.DataFrame
    ) -> float:
        """
        Predict the next candle return given historical OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data (must have at least seq_len rows)
            
        Returns:
            Predicted return for next candle
        """
        if len(df) < self.seq_len:
            raise ValueError(
                f"Need at least {self.seq_len} candles, got {len(df)}"
            )
        
        # Use last seq_len candles
        df_recent = df.iloc[-self.seq_len:].copy()
        
        # Engineer features
        features = self.fe.create_features(df_recent, normalize=True)
        
        # Convert to tensor
        x = torch.FloatTensor(features.values).unsqueeze(0)  # (1, seq_len, n_features)
        x = x.to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(x)
        
        return prediction.item()
    
    def predict_batch(
        self,
        df_list: List[pd.DataFrame]
    ) -> np.ndarray:
        """
        Predict returns for a batch of sequences.
        
        Args:
            df_list: List of DataFrames with OHLCV data
            
        Returns:
            Array of predicted returns
        """
        batch_features = []
        
        for df in df_list:
            if len(df) < self.seq_len:
                raise ValueError(
                    f"All sequences must have at least {self.seq_len} candles"
                )
            
            df_recent = df.iloc[-self.seq_len:].copy()
            features = self.fe.create_features(df_recent, normalize=True)
            batch_features.append(features.values)
        
        # Convert to tensor
        x = torch.FloatTensor(np.array(batch_features))  # (batch, seq_len, n_features)
        x = x.to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(x)
        
        return predictions.cpu().numpy().squeeze()
    
    def predict_with_confidence(
        self,
        df: pd.DataFrame,
        n_samples: int = 100,
        dropout_rate: float = 0.1
    ) -> Tuple[float, float]:
        """
        Predict with uncertainty estimation using Monte Carlo dropout.
        
        Args:
            df: DataFrame with OHLCV data
            n_samples: Number of MC samples
            dropout_rate: Dropout rate for MC sampling
            
        Returns:
            mean_prediction, std_prediction
        """
        if len(df) < self.seq_len:
            raise ValueError(
                f"Need at least {self.seq_len} candles, got {len(df)}"
            )
        
        df_recent = df.iloc[-self.seq_len:].copy()
        features = self.fe.create_features(df_recent, normalize=True)
        
        x = torch.FloatTensor(features.values).unsqueeze(0)
        x = x.to(self.device)
        
        # Enable dropout for uncertainty estimation
        self.model.train()
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.model(x)
                predictions.append(pred.item())
        
        # Disable dropout again
        self.model.eval()
        
        predictions = np.array(predictions)
        return predictions.mean(), predictions.std()
    
    def backtest(
        self,
        df: pd.DataFrame,
        start_idx: int = None
    ) -> pd.DataFrame:
        """
        Backtest predictions on historical data.
        
        Args:
            df: Full DataFrame with OHLCV data
            start_idx: Index to start backtesting (default: seq_len)
            
        Returns:
            DataFrame with predictions and actual returns
        """
        if start_idx is None:
            start_idx = self.seq_len
        
        predictions = []
        actuals = []
        timestamps = []
        
        for i in range(start_idx, len(df) - 1):
            # Get historical data up to current point
            historical_data = df.iloc[:i+1]
            
            # Predict
            pred = self.predict_next_return(historical_data)
            predictions.append(pred)
            
            # Get actual return
            actual = (df.iloc[i+1]['close'] - df.iloc[i]['close']) / df.iloc[i]['close']
            actuals.append(actual)
            
            timestamps.append(df.index[i])
        
        # Create results DataFrame
        results = pd.DataFrame({
            'timestamp': timestamps,
            'predicted_return': predictions,
            'actual_return': actuals,
            'error': np.array(predictions) - np.array(actuals),
            'abs_error': np.abs(np.array(predictions) - np.array(actuals))
        })
        
        # Calculate metrics
        results['correct_direction'] = (
            np.sign(results['predicted_return']) == np.sign(results['actual_return'])
        )
        
        return results


def load_trained_model(
    checkpoint_path: str,
    input_dim: int,
    device: torch.device
) -> TimeSeriesMambaModel:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        input_dim: Input feature dimension
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    # Create model with same architecture
    model = TimeSeriesMambaModel(
        input_dim=input_dim,
        d_model=256,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand_factor=2,
        n_heads=8,
        mlp_hidden_dim=512,
        dropout=0.1
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    return model


def main():
    """Demo inference script."""
    print("=" * 80)
    print("Time-Series Equity Prediction - Inference Demo")
    print("=" * 80)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = 200
    
    # Create or load test data
    print("\nCreating test data...")
    from initPreprocess import create_sample_data
    df = create_sample_data(n_samples=1000, seed=123)
    print(f"  Data shape: {df.shape}")
    
    # Check if trained model exists
    checkpoint_path = './checkpoints/best_model.pt'
    
    if not Path(checkpoint_path).exists():
        print("\n⚠️  No trained model found!")
        print("  Please run train.py first to train a model.")
        print("  Using a fresh model for demonstration purposes only.")
        
        # Create fresh model for demo
        fe = FeatureEngineering()
        features = fe.create_features(df.iloc[:seq_len], normalize=True)
        input_dim = features.shape[1]
        
        model = TimeSeriesMambaModel(
            input_dim=input_dim,
            d_model=256,
            n_layers=4
        )
    else:
        print(f"\n✓ Loading trained model from {checkpoint_path}")
        
        # Create feature engineer
        fe = FeatureEngineering()
        features = fe.create_features(df.iloc[:seq_len], normalize=True)
        input_dim = features.shape[1]
        
        # Load model
        model = load_trained_model(checkpoint_path, input_dim, device)
        
        # Load metrics
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'metrics' in checkpoint:
            print("\n  Model Performance:")
            for metric_name, metric_value in checkpoint['metrics'].items():
                print(f"    {metric_name}: {metric_value:.4f}")
    
    # Create predictor
    print("\nInitializing predictor...")
    predictor = TimeSeriesPredictor(
        model=model,
        feature_engineer=fe,
        device=device,
        seq_len=seq_len
    )
    
    # Single prediction
    print("\n" + "-" * 80)
    print("Single Prediction Example")
    print("-" * 80)
    
    historical_data = df.iloc[:seq_len + 100]
    prediction = predictor.predict_next_return(historical_data)
    
    actual_return = (
        df.iloc[seq_len + 100]['close'] - df.iloc[seq_len + 99]['close']
    ) / df.iloc[seq_len + 99]['close']
    
    print(f"  Current price: ${df.iloc[seq_len + 99]['close']:.2f}")
    print(f"  Predicted return: {prediction:.4%}")
    print(f"  Actual return: {actual_return:.4%}")
    print(f"  Error: {abs(prediction - actual_return):.4%}")
    print(f"  Direction correct: {np.sign(prediction) == np.sign(actual_return)}")
    
    # Prediction with confidence
    print("\n" + "-" * 80)
    print("Prediction with Uncertainty")
    print("-" * 80)
    
    mean_pred, std_pred = predictor.predict_with_confidence(
        historical_data,
        n_samples=50
    )
    
    print(f"  Mean prediction: {mean_pred:.4%}")
    print(f"  Std deviation: {std_pred:.4%}")
    print(f"  95% CI: [{mean_pred - 1.96*std_pred:.4%}, {mean_pred + 1.96*std_pred:.4%}]")
    
    # Backtesting
    print("\n" + "-" * 80)
    print("Backtesting on Historical Data")
    print("-" * 80)
    
    # Backtest on last 50 points
    backtest_data = df.iloc[:seq_len + 50]
    results = predictor.backtest(backtest_data, start_idx=seq_len)
    
    print(f"\n  Backtest Results ({len(results)} predictions):")
    print(f"    Mean Absolute Error: {results['abs_error'].mean():.4%}")
    print(f"    RMSE: {np.sqrt((results['error']**2).mean()):.4%}")
    print(f"    Direction Accuracy: {results['correct_direction'].mean():.2%}")
    print(f"    Correlation: {results['predicted_return'].corr(results['actual_return']):.4f}")
    
    print("\n  Sample Predictions:")
    print(results[['predicted_return', 'actual_return', 'correct_direction']].head(10))
    
    # Batch prediction
    print("\n" + "-" * 80)
    print("Batch Prediction Example")
    print("-" * 80)
    
    # Create batch of sequences
    batch_dfs = [
        df.iloc[i:i+seq_len] for i in range(0, 100, 20)
    ]
    
    batch_predictions = predictor.predict_batch(batch_dfs)
    
    print(f"  Predicted {len(batch_predictions)} returns:")
    for i, pred in enumerate(batch_predictions):
        print(f"    Sequence {i+1}: {pred:.4%}")
    
    print("\n" + "=" * 80)
    print("Inference demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()