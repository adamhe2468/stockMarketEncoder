"""
Complete Training Pipeline for Time-Series Equity Prediction Model
Integrates preprocessing, model architecture, and training loop
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, Optional, Tuple
import time
from pathlib import Path
from tqdm import tqdm

# Import custom modules
#from modlPred import TimeSeriesMambaModel
#from dataLoader import MultiTickerDataLoader


# Removed PyTorchTimeSeriesDataset - using MultiTickerDataLoader instead


class Trainer:
    """
    Training manager for time-series prediction model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        clip_grad_norm: float = 1.0,
        log_dir: str = './runs'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.clip_grad_norm = clip_grad_norm

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

        # TensorBoard logging
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"  TensorBoard logging to: {log_dir}")
    
    def train_epoch(self, epoch: int, num_epochs: int) -> float:
        """Train for one epoch with progress bar."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Progress bar for training
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{num_epochs}",
                    ncols=100, leave=True)

        for features, targets in pbar:
            # Move to device
            features = features.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            predictions = self.model(features)
            loss = self.criterion(predictions, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.clip_grad_norm
                )

            # Optimizer step
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar with current loss
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'loss': f'{avg_loss:.6f}'})

        return avg_loss
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in self.val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(features)
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # Calculate metrics
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        metrics = self._calculate_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        # MSE (already have from loss)
        mse = np.mean((predictions - targets) ** 2)
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # MAE
        mae = np.mean(np.abs(predictions - targets))
        
        # R² Score
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        # Direction accuracy (did we predict the right direction?)
        pred_direction = np.sign(predictions)
        true_direction = np.sign(targets)
        direction_accuracy = np.mean(pred_direction == true_direction)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy
        }
    
    def train(
        self,
        num_epochs: int,
        save_dir: str = './checkpoints',
        verbose: bool = True
    ):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            verbose: Print training progress
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch, num_epochs)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, metrics = self.validate()
            self.val_losses.append(val_loss)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log to TensorBoard (per epoch)
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Metrics/RMSE', metrics['rmse'], epoch)
            self.writer.add_scalar('Metrics/MAE', metrics['mae'], epoch)
            self.writer.add_scalar('Metrics/MSE', metrics['mse'], epoch)
            self.writer.add_scalar('Metrics/R2', metrics['r2'], epoch)
            self.writer.add_scalar('Metrics/Direction_Accuracy', metrics['direction_accuracy'], epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    f"{save_dir}/best_model.pt",
                    epoch,
                    val_loss,
                    metrics
                )

            # Print progress (Keras-style)
            if verbose:
                epoch_time = time.time() - start_time
                print(f" - {epoch_time:.0f}s - "
                      f"loss: {train_loss:.4f} - "
                      f"val_loss: {val_loss:.4f} - "
                      f"rmse: {metrics['rmse']:.4f} - "
                      f"mae: {metrics['mae']:.4f} - "
                      f"r2: {metrics['r2']:.4f} - "
                      f"dir_acc: {metrics['direction_accuracy']:.4f} - "
                      f"lr: {current_lr:.2e}")

        # Close TensorBoard writer
        self.writer.close()
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_loss: float,
        metrics: Dict[str, float]
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        return checkpoint


# Removed prepare_data - using MultiTickerDataLoader instead


def main():
    """Main training script."""
    print("=" * 80)
    print("Time-Series Equity Prediction - Training Pipeline")
    print("=" * 80)
    
    # Configuration
    CONFIG = {
        # Data
        'processed_data_dir': 'processed_data',
        'batch_size': 256,  # For large dataset (34.4M sequences)
        'num_workers': 0,

        # Model Architecture (Large model for pre-training: ~1.5M+ parameters)
        'd_model': 256,   # Increased embedding dimension
        'n_layers': 6,    # More Mamba layers for depth
        'd_state': 16,    # Increased state dimension
        'd_conv': 4,
        'expand_factor': 2,
        'n_heads': 8,     # More attention heads
        'mlp_hidden_dim': 512,  # Larger MLP
        'dropout': 0.2,   # Lower dropout for pre-training (can increase for fine-tuning)

        # Training (Pre-training phase)
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,  # Increased from 1e-5 for stronger regularization
        'clip_grad_norm': 1.0,

        # Other
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Set random seeds
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])

    # Load preprocessed data
    print("\n" + "-" * 80)
    print("Loading preprocessed data...")
    data_loader = MultiTickerDataLoader(
        processed_data_dir=CONFIG['processed_data_dir'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers']
    )

    # Get dataset info
    info = data_loader.get_dataset_info()
    print(f"  Tickers loaded: {info['n_tickers']}")
    print(f"  Input dimension: {info['n_features']}")
    print(f"  Sequence length: {info['seq_len']}")
    # Create DataLoaders
    print("\n" + "-" * 80)
    print("Creating DataLoaders...")
    train_loader, val_loader, test_loader = data_loader.get_dataloaders()
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Set input_dim from data
    input_dim = info['n_features']  # Should be 18
    
    # Create model
    print("\n" + "-" * 80)
    print("Creating model...")
    device = torch.device(CONFIG['device'])
    
    model = TimeSeriesMambaModel(
        input_dim=input_dim,
        d_model=CONFIG['d_model'],
        n_layers=CONFIG['n_layers'],
        d_state=CONFIG['d_state'],
        d_conv=CONFIG['d_conv'],
        expand_factor=CONFIG['expand_factor'],
        n_heads=CONFIG['n_heads'],
        mlp_hidden_dim=CONFIG['mlp_hidden_dim'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    print(f"  Model parameters: {model.count_parameters():,}")
    print(f"  Device: {device}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        clip_grad_norm=CONFIG['clip_grad_norm']
    )
    
    # Train
    print("\n" + "-" * 80)
    print("Training...")
    print("-" * 80)
    
    trainer.train(
        num_epochs=CONFIG['num_epochs'],
        save_dir='./checkpoints',
        verbose=True
    )
    
    # Final results
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"Final train loss: {trainer.train_losses[-1]:.6f}")
    print(f"Final val loss: {trainer.val_losses[-1]:.6f}")

    # Save final model
    trainer.save_checkpoint(
        './checkpoints/final_model.pt',
        CONFIG['num_epochs'],
        trainer.val_losses[-1],
        {}
    )
    print("\nModel saved to ./checkpoints/")

    # =========================================================================
    # TRADING PERFORMANCE ANALYSIS (MSE alone is meaningless!)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Trading Performance Analysis on Test Set")
    print("=" * 80)

    model.eval()
    all_predictions = []
    all_targets = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            predictions = model(features)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    predictions = np.concatenate(all_predictions).flatten()
    actual = np.concatenate(all_targets).flatten()

    # Trading Strategy Analysis
    print("\n" + "-" * 80)
    print("1. DIRECTIONAL ACCURACY")
    print("-" * 80)

    # Directional accuracy
    pred_direction = np.sign(predictions)
    actual_direction = np.sign(actual)
    direction_correct = (pred_direction == actual_direction).sum()
    direction_accuracy = direction_correct / len(predictions) * 100

    print(f"  Overall Direction Accuracy: {direction_accuracy:.2f}%")
    print(f"  Correct predictions: {direction_correct:,} / {len(predictions):,}")

    # Long-only accuracy
    long_mask = predictions > 0
    if long_mask.sum() > 0:
        long_correct = ((predictions[long_mask] * actual[long_mask]) > 0).sum()
        long_accuracy = long_correct / long_mask.sum() * 100
        print(f"  Long-only Accuracy: {long_accuracy:.2f}% ({long_mask.sum():,} signals)")

    # Short-only accuracy
    short_mask = predictions < 0
    if short_mask.sum() > 0:
        short_correct = ((predictions[short_mask] * actual[short_mask]) > 0).sum()
        short_accuracy = short_correct / short_mask.sum() * 100
        print(f"  Short-only Accuracy: {short_accuracy:.2f}% ({short_mask.sum():,} signals)")

    # Confidence-based accuracy (top 10%, 20%, 50%)
    print("\n" + "-" * 80)
    print("2. CONFIDENCE-BASED ACCURACY")
    print("-" * 80)

    abs_predictions = np.abs(predictions)
    for percentile in [90, 80, 50]:
        threshold = np.percentile(abs_predictions, percentile)
        confident_mask = abs_predictions >= threshold
        if confident_mask.sum() > 0:
            confident_correct = (pred_direction[confident_mask] == actual_direction[confident_mask]).sum()
            confident_accuracy = confident_correct / confident_mask.sum() * 100
            print(f"  Top {100-percentile}% Confident: {confident_accuracy:.2f}% "
                  f"({confident_mask.sum():,} trades)")

    # Simulated trading performance
    print("\n" + "-" * 80)
    print("3. SIMULATED TRADING RETURNS")
    print("-" * 80)

    # Simple strategy: trade in direction of prediction
    returns_if_traded = predictions * actual  # Positive = profit, Negative = loss

    # All trades
    total_return = returns_if_traded.sum() * 100
    avg_return_per_trade = returns_if_traded.mean() * 100
    win_rate = (returns_if_traded > 0).sum() / len(returns_if_traded) * 100

    print(f"  Total Cumulative Return: {total_return:.2f}%")
    print(f"  Average Return per Trade: {avg_return_per_trade:.4f}%")
    print(f"  Win Rate: {win_rate:.2f}%")

    # Top 20% confident trades
    top20_mask = abs_predictions >= np.percentile(abs_predictions, 80)
    if top20_mask.sum() > 0:
        top20_returns = returns_if_traded[top20_mask].sum() * 100
        top20_avg = returns_if_traded[top20_mask].mean() * 100
        top20_win_rate = (returns_if_traded[top20_mask] > 0).sum() / top20_mask.sum() * 100
        print(f"\n  Top 20% Confident Trades:")
        print(f"    Cumulative Return: {top20_returns:.2f}%")
        print(f"    Average Return: {top20_avg:.4f}%")
        print(f"    Win Rate: {top20_win_rate:.2f}%")

    # Risk metrics
    print("\n" + "-" * 80)
    print("4. RISK METRICS")
    print("-" * 80)

    # Sharpe-like ratio (returns / volatility)
    returns_std = returns_if_traded.std() * 100
    sharpe_like = (avg_return_per_trade / returns_std) if returns_std > 0 else 0

    # Max drawdown simulation
    cumulative_returns = np.cumsum(returns_if_traded) * 100
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = running_max - cumulative_returns
    max_drawdown = drawdown.max()

    print(f"  Returns Volatility: {returns_std:.4f}%")
    print(f"  Sharpe-like Ratio: {sharpe_like:.4f}")
    print(f"  Max Drawdown: {max_drawdown:.2f}%")

    # Prediction distribution
    print("\n" + "-" * 80)
    print("5. PREDICTION DISTRIBUTION")
    print("-" * 80)

    print(f"  Prediction Mean: {predictions.mean():.6f}")
    print(f"  Prediction Std: {predictions.std():.6f}")
    print(f"  Prediction Range: [{predictions.min():.6f}, {predictions.max():.6f}]")
    print(f"  Actual Mean: {actual.mean():.6f}")
    print(f"  Actual Std: {actual.std():.6f}")

    # Correlation
    correlation = np.corrcoef(predictions, actual)[0, 1]
    print(f"\n  Prediction-Actual Correlation: {correlation:.4f}")

    # MSE vs R² Comparison
    print("\n" + "-" * 80)
    print("6. MSE vs R² COMPARISON (Understanding the metrics)")
    print("-" * 80)

    # Calculate MSE and RMSE
    mse = np.mean((predictions - actual) ** 2)
    rmse = np.sqrt(mse)

    # Calculate R²
    ss_res = np.sum((actual - predictions) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f} (= {rmse*100:.3f}% average error)")
    print(f"  R²: {r2:.6f}")

    # Interpretation
    print(f"\n  Interpretation:")
    if r2 < 0:
        print(f"    R² < 0: Model worse than predicting mean (BAD!)")
    elif r2 < 0.01:
        print(f"    R² < 0.01: Explains <1% variance (essentially random)")
    elif r2 < 0.05:
        print(f"    R² < 0.05: Explains ~{r2*100:.1f}% variance (weak but possible signal)")
    elif r2 < 0.15:
        print(f"    R² < 0.15: Explains ~{r2*100:.1f}% variance (GOOD for stocks!)")
    else:
        print(f"    R² = {r2:.4f}: Explains {r2*100:.1f}% variance (EXCELLENT - verify no look-ahead!)")

    print(f"\n  Key Insight:")
    print(f"    MSE tells you: Average squared error magnitude")
    print(f"    R² tells you: How much variance you explain")
    print(f"    Direction Accuracy tells you: Can you make money?")
    print(f"\n  For stock trading:")
    print(f"    - Direction Acc > 52% is more important than low MSE")
    print(f"    - R² > 0.05 is already decent for daily returns")
    print(f"    - R² > 0.15 would be exceptional (rare in practice)")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print("\nNOTE: These are in-sample test results. Real performance will differ!")
    print("      Always validate with out-of-sample data and forward testing.")
    print("=" * 80)


if __name__ == "__main__":
    main()