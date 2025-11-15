"""
Time-Series Prediction Model with Mamba Encoder and Cross-Attention Decoder
Architecture for equity price movement prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
import torch.func as func

BIAS_SCALE_FACTOR = 2.7 
class SimplifiedMambaBlock(nn.Module):
    """
    Simplified Mamba-style State Space Model block.
    
    Implements a selective state-space model with:
    - Input projection
    - Selective scan mechanism (simplified SSM)
    - Gating mechanism
    - Output projection with residual connection
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Convolution kernel size for temporal mixing
            expand_factor: Expansion factor for inner dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand_factor
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)  # For input and gate
        
        # Convolution for local temporal mixing
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner  # Depthwise convolution
        )
        
        # SSM parameters (simplified)
        # In real Mamba, these would be dynamically generated based on input
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        self.D = nn.Parameter(torch.randn(self.d_inner))
        
        # Selective mechanism: projects input to SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)  # For B and C matrices
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Normalization and regularization
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Mamba block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # Layer norm
        x = self.norm(x)
        
        # Input projection: split into main path and gate
        x_and_gate = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x_main, gate = x_and_gate.chunk(2, dim=-1)  # Each: (batch, seq_len, d_inner)
        
        # Temporal mixing via convolution
        # Conv1d expects (batch, channels, seq_len)
        x_main_transposed = x_main.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_main_transposed)[:, :, :seq_len]  # Trim padding
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_inner)
        x_conv = F.silu(x_conv)  # Activation
        
        # Selective SSM mechanism
        # Generate B and C matrices from input (selective mechanism)
        ssm_params = self.x_proj(x_conv)  # (batch, seq_len, d_state * 2)
        B, C = ssm_params.chunk(2, dim=-1)  # Each: (batch, seq_len, d_state)
        
        # Simplified SSM scan (discretized state-space model)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Simplified selective scan
        # In real Mamba, this would be a more sophisticated parallel scan
        y = self._selective_scan(x_conv, A, B, C)  # (batch, seq_len, d_inner)
        
        # Add skip connection (D parameter)
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        
        # Gating mechanism
        y = y * F.silu(gate)
        
        # Output projection
        output = self.out_proj(y)  # (batch, seq_len, d_model)
        output = self.dropout(output)
        
        # Residual connection
        return output + residual
    
    def _selective_scan(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """
        Simplified selective scan operation.
        
        In a full Mamba implementation, this would use a parallel scan algorithm.
        Here we use a simplified recurrent approach for clarity.
        
        Args:
            x: Input (batch, seq_len, d_inner)
            A: State transition matrix (d_inner, d_state)
            B: Input matrix (batch, seq_len, d_state)
            C: Output matrix (batch, seq_len, d_state)
            
        Returns:
            Output (batch, seq_len, d_inner)
        """
        batch_size, seq_len, d_inner = x.shape
        d_state = B.shape[-1]
        
        # Initialize state
        h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        # Discretization step (simplified)
        deltaA = torch.exp(A.unsqueeze(0))  # (1, d_inner, d_state)
        
        for t in range(seq_len):
            # Current input and parameters
            x_t = x[:, t, :]  # (batch, d_inner)
            B_t = B[:, t, :]  # (batch, d_state)
            C_t = C[:, t, :]  # (batch, d_state)
            
            # State update: h = A * h + B * x
            h = deltaA * h + torch.einsum('bi,bj->bij', x_t, B_t)
            
            # Output: y = C * h
            y_t = torch.einsum('bij,bj->bi', h, C_t)  # (batch, d_inner)
            outputs.append(y_t)
        
        # Stack outputs
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)
        
        return y


class MambaEncoder(nn.Module):
    """
    Encoder stack with multiple Mamba blocks.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Input feature dimension
            d_model: Model hidden dimension
            n_layers: Number of Mamba blocks
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand_factor: Expansion factor for Mamba blocks
            dropout: Dropout rate
        """
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            SimplifiedMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Encoded sequence of shape (batch, seq_len, d_model)
        """
        # Project input to model dimension
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Pass through Mamba blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.norm(x)
        
        return x


class CrossAttentionDecoder(nn.Module):
    """
    Cross-attention decoder with learnable query for next-step prediction.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Learnable query for prediction (what to predict next)
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model))
        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, encoder_output: torch.Tensor, dynamic_bias: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention over encoder outputs.

        Args:
            encoder_output: Encoder outputs of shape (batch, seq_len, d_model)
            dynamic_bias: Per-timestep bias of shape (batch, 1, seq_len) for attention modulation

        Returns:
            Context vector of shape (batch, d_model)
        """
        batch_size, seq_len, _ = encoder_output.shape
        
        # Expand query token for batch
        query = self.query_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        
        # Project Q, K, V
        Q = self.q_proj(query)  # (batch, 1, d_model)
        K = self.k_proj(encoder_output)  # (batch, seq_len, d_model)
        V = self.v_proj(encoder_output)  # (batch, seq_len, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)
        # (batch, n_heads, 1, head_dim)
        
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # (batch, n_heads, seq_len, head_dim)
        
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # (batch, n_heads, seq_len, head_dim)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # (batch, n_heads, 1, seq_len)

        # Add per-timestep dynamic bias
        # Reshape bias from (batch, 1, seq_len) to (batch, 1, 1, seq_len) for broadcasting
        dynamic_bias = dynamic_bias.unsqueeze(1)  # (batch, 1, 1, seq_len)
        attn_scores = attn_scores + dynamic_bias * BIAS_SCALE_FACTOR
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (batch, n_heads, 1, head_dim)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, 1, n_heads, head_dim)
        attn_output = attn_output.view(batch_size, 1, self.d_model)  # (batch, 1, d_model)
        
        output = self.out_proj(attn_output)  # (batch, 1, d_model)
        output = self.dropout(output)
        
        # Add residual and normalize
        output = self.norm(output + query)
        
        # Squeeze to get (batch, d_model)
        output = output.squeeze(1)
        
        return output

class subnetMlp(nn.Module):
    """
    Subnet MLP that generates per-timestep dynamic bias from encoder output.

    Applies an MLP independently to each timestep in the sequence to produce
    position-dependent attention bias values in the range [-1, 1].
    """

    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Input dimension (matches encoder output dimension)
            hidden_dim: Hidden layer dimension for the MLP
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
    
    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, seq_len, d_model)
        Returns:
            Per-timestep bias: (batch, 1, seq_len)
        """
        batch, seq_len, d_model = encoder_output.shape
        
        # Reshape: (batch * seq_len, d_model)
        x = encoder_output.reshape(batch * seq_len, d_model)
        
        # Apply net to each timestep
        bias = self.net(x)  # (batch * seq_len, 1)
        
        # Reshape: (batch, seq_len, 1)
        bias = bias.reshape(batch, seq_len, 1)
        
        # Transpose for attention: (batch, 1, seq_len)
        bias = bias.transpose(1, 2)
        
        return bias

class MLPHead(nn.Module):
    """
    MLP regression head for scalar prediction.
    """
    
    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Input dimension
            hidden_dim: Hidden layer dimension (default: d_model * 2)
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = d_model * 2
        
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Output: scalar prediction
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP head.
        
        Args:
            x: Input tensor of shape (batch, d_model)
            
        Returns:
            Predictions of shape (batch, 1)
        """
        return self.net(x)


class TimeSeriesMambaModel(nn.Module):
    """
    Complete time-series prediction model with Mamba encoder and cross-attention decoder.
    
    Architecture:
        1. MambaEncoder: Processes the input sequence with state-space models
        2. CrossAttentionDecoder: Attends over encoder outputs with learnable query
        3. MLPHead: Produces scalar prediction (next candle return)
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand_factor: int = 2,
        n_heads: int = 8,
        mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Number of input features per time step
            d_model: Model hidden dimension
            n_layers: Number of Mamba blocks in encoder
            d_state: SSM state dimension
            d_conv: Convolution kernel size for Mamba
            expand_factor: Expansion factor for Mamba blocks
            n_heads: Number of attention heads in decoder
            mlp_hidden_dim: Hidden dimension for MLP head
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Encoder: Mamba-based sequence encoder
        self.encoder = MambaEncoder(
            input_dim=input_dim,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand_factor,
            dropout=dropout
        )
        self.subnetMlp = subnetMlp(d_model=d_model, hidden_dim=128, dropout=dropout)
            
        # Decoder: Cross-attention over encoder outputs
        self.decoder = CrossAttentionDecoder(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        # Head: MLP for regression
        self.head = MLPHead(
            d_model=d_model,
            hidden_dim=mlp_hidden_dim,
            dropout=dropout
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
               seq_len should be 200 in production
               
        Returns:
            Predictions of shape (batch_size, 1) - next candle return
        """
        # Encode sequence
        encoder_output = self.encoder(x)  # (batch, seq_len, d_model)
        subnetMlp_output = self.subnetMlp(encoder_output) #tanh 
        # Decode with cross-attention
        context = self.decoder(encoder_output, subnetMlp_output)  # (batch, d_model)
        
        # Predict next candle return
        prediction = self.head(context)  # (batch, 1)
        
        return prediction
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("Time-Series Mamba Model - Architecture Test")
    print("=" * 80)
    
    # Model configuration
    INPUT_DIM = 32  # Number of features per time step
    SEQ_LEN = 200   # Sequence length
    BATCH_SIZE = 8
    D_MODEL = 256
    N_LAYERS = 4
    
    # Create model
    model = TimeSeriesMambaModel(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        d_state=16,
        d_conv=4,
        expand_factor=2,
        n_heads=8,
        mlp_hidden_dim=512,
        dropout=0.1
    )
    
    print(f"\nModel Architecture:")
    print(f"  Input Dimension: {INPUT_DIM}")
    print(f"  Model Dimension: {D_MODEL}")
    print(f"  Number of Layers: {N_LAYERS}")
    print(f"  Total Parameters: {model.count_parameters():,}")
    
    # Create dummy input
    # Shape: (batch_size, seq_len, input_dim)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    print(f"\nInput Shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    print(f"Output Shape: {output.shape}")
    
    # Create dummy targets (next candle returns)
    # Shape: (batch_size, 1)
    targets = torch.randn(BATCH_SIZE, 1) * 0.02  # ±2% returns
    print(f"Target Shape: {targets.shape}")
    
    # Training example
    print("\n" + "=" * 80)
    print("Training Example")
    print("=" * 80)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Training step
    model.train()
    
    # Forward pass
    predictions = model(x)
    
    # Compute loss
    loss = criterion(predictions, targets)
    
    print(f"\nBefore optimization:")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Sample predictions: {predictions[:3].squeeze().tolist()}")
    print(f"  Sample targets: {targets[:3].squeeze().tolist()}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping (recommended for training stability)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Optimizer step
    optimizer.step()
    
    # Forward pass after update
    with torch.no_grad():
        predictions_after = model(x)
        loss_after = criterion(predictions_after, targets)
    
    print(f"\nAfter one optimization step:")
    print(f"  Loss: {loss_after.item():.6f}")
    print(f"  Loss change: {loss_after.item() - loss.item():.6f}")
    
    print("\n" + "=" * 80)
    print("Architecture test completed successfully!")
    print("=" * 80)