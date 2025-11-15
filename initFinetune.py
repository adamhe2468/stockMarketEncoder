"""
Fine-tuning Script for Frozen Encoder Transfer Learning
Loads pre-trained encoder and trains only the prediction head on small datasets
"""

import torch
import torch.nn as nn
from pathlib import Path
from initTrain import Trainer
from modlPred import TimeSeriesMambaModel
from dataLoader import MultiTickerDataLoader


def freeze_encoder(model: TimeSeriesMambaModel):
    """
    Freeze the encoder stack (Mamba + subnetMlp + CrossAttention).
    Only the final prediction head (mlp_head) will be trainable.
    """
    print("Freezing encoder components...")

    # Freeze input projection
    for param in model.input_proj.parameters():
        param.requires_grad = False

    # Freeze all Mamba layers
    for layer in model.mamba_layers:
        for param in layer.parameters():
            param.requires_grad = False

    # Freeze subnetMlp (dynamic bias generator)
    for param in model.subnet_mlp.parameters():
        param.requires_grad = False

    # Freeze cross-attention decoder
    for param in model.decoder.parameters():
        param.requires_grad = False

    # Keep prediction head trainable
    for param in model.mlp_head.parameters():
        param.requires_grad = True

    # Count trainable vs frozen parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"  Trainable parameters: {trainable:,} ({trainable/(trainable+frozen)*100:.1f}%)")
    print(f"  Frozen parameters: {frozen:,} ({frozen/(trainable+frozen)*100:.1f}%)")

    return model


def main():
    """Fine-tuning script for small datasets."""
    print("=" * 80)
    print("Fine-Tuning with Frozen Encoder - Transfer Learning")
    print("=" * 80)

    # Configuration for fine-tuning
    CONFIG = {
        # Pre-trained model checkpoint
        'pretrained_checkpoint': './checkpoints/best_model.pt',

        # Data (use small dataset directory)
        'processed_data_dir': 'processed_data_finetune',  # Small dataset
        'batch_size': 32,  # Smaller batch for small dataset
        'num_workers': 0,

        # Fine-tuning settings
        'freeze_encoder': True,  # Freeze Mamba + subnetMlp + CrossAttention
        'num_epochs': 100,  # Many epochs with early stopping
        'learning_rate': 1e-5,  # Very low LR for fine-tuning
        'weight_decay': 1e-2,  # Very strong L2 regularization
        'clip_grad_norm': 0.3,  # Tight gradient clipping
        'dropout': 0.5,  # Very high dropout (will be ignored if loading pretrained)

        # Early stopping
        'early_stopping_patience': 10,

        # Other
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("\nFine-tuning Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    # Set random seeds
    torch.manual_seed(CONFIG['seed'])

    # Load small dataset
    print("\n" + "-" * 80)
    print("Loading fine-tuning dataset...")
    data_loader = MultiTickerDataLoader(
        processed_data_dir=CONFIG['processed_data_dir'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers']
    )

    info = data_loader.get_dataset_info()
    print(f"  Tickers: {info['n_tickers']}")
    print(f"  Input dim: {info['n_features']}")
    print(f"  Train samples: {info['n_train']:,}")
    print(f"  Val samples: {info['n_val']:,}")

    train_loader, val_loader, test_loader = data_loader.get_dataloaders()

    # Load pre-trained model
    print("\n" + "-" * 80)
    print("Loading pre-trained model...")
    device = torch.device(CONFIG['device'])

    checkpoint = torch.load(CONFIG['pretrained_checkpoint'], map_location=device)

    # Get model config from checkpoint (or infer from saved state_dict)
    # Must match pre-training architecture exactly!
    model = TimeSeriesMambaModel(
        input_dim=info['n_features'],
        d_model=64,   # Must match pre-training config
        n_layers=2,
        d_state=8,
        d_conv=4,
        expand_factor=2,
        n_heads=2,
        mlp_hidden_dim=128,
        dropout=CONFIG['dropout']
    ).to(device)

    # Load pre-trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded from epoch {checkpoint['epoch']}")
    print(f"  Pre-trained val loss: {checkpoint['val_loss']:.6f}")

    # Freeze encoder if specified
    if CONFIG['freeze_encoder']:
        model = freeze_encoder(model)

    # Setup optimizer (only for trainable parameters)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # Scheduler with early stopping in mind
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Loss - Using SmoothL1Loss (Huber loss) for robustness to outliers
    criterion = nn.SmoothL1Loss(beta=0.1)  # More robust than MSE for small datasets

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        clip_grad_norm=CONFIG['clip_grad_norm'],
        log_dir='./runs_finetune'
    )

    # Train with early stopping
    print("\n" + "-" * 80)
    print("Fine-tuning...")
    print("-" * 80)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, CONFIG['num_epochs'] + 1):
        # Manual training loop with early stopping
        train_loss = trainer.train_epoch(epoch, CONFIG['num_epochs'])
        val_loss, metrics = trainer.validate()

        # Logging
        trainer.writer.add_scalar('Loss/train', train_loss, epoch)
        trainer.writer.add_scalar('Loss/val', val_loss, epoch)
        trainer.writer.add_scalar('Metrics/RMSE', metrics['rmse'], epoch)
        trainer.writer.add_scalar('Metrics/MAE', metrics['mae'], epoch)
        trainer.writer.add_scalar('Metrics/R2', metrics['r2'], epoch)
        trainer.writer.add_scalar('Metrics/Direction_Accuracy', metrics['direction_accuracy'], epoch)

        # Scheduler step
        scheduler.step(val_loss)

        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f" - loss: {train_loss:.4f} - "
              f"val_loss: {val_loss:.4f} - "
              f"rmse: {metrics['rmse']:.4f} - "
              f"mae: {metrics['mae']:.4f} - "
              f"r2: {metrics['r2']:.4f} - "
              f"dir_acc: {metrics['direction_accuracy']:.4f} - "
              f"lr: {current_lr:.2e}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics
            }, './checkpoints/best_finetuned_model.pt')
            print(f"  → Best model saved! (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{CONFIG['early_stopping_patience']})")

            if patience_counter >= CONFIG['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch} epochs!")
                break

    trainer.writer.close()

    # Final results
    print("\n" + "=" * 80)
    print("Fine-tuning Complete!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: ./checkpoints/best_finetuned_model.pt")


if __name__ == "__main__":
    main()
