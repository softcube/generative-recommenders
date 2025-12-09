from __future__ import annotations

import math
import copy
import warnings
from dataclasses import dataclass
from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

from apps.training_torch.src.training_blocks.blocks import (
    MANAGER,
    Block,
    MLPExperts,
)
from apps.training_torch.src.training_blocks.kv_cache import KVCache
from apps.training_torch.src.training_blocks.mamba_block_minimal import (
    MambaBlockMinimal,
    ModelArgsMinimal,
)
from apps.training_torch.src.training_layers.custom import (
    LayerNorm,
    NewGELU,
    L2Norm,
    RMSNorm,
)
from pkg.constants import DEFAULT_RANDOM_SEED

# Suppress torch._dynamo errors globally for LoRA compatibility
# This helps prevent compilation issues with custom LoRA operations
try:
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True
except (ImportError, AttributeError):
    # torch._dynamo not available in older PyTorch versions
    pass


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def infer_device_from_embedding(embedding_module, model=None):
    """Infer the device for an embedding-like module in a robust way.

    Supports plain `nn.Embedding`, adapter-wrapped modules (with nested
    `.wte.weight`), or falls back to the first model parameter.
    """
    # Fast-path: direct weight on the module
    if hasattr(embedding_module, "weight") and isinstance(embedding_module.weight, torch.Tensor):
        return embedding_module.weight.device

    # Adapter path: nested `.wte.weight`
    if (
        hasattr(embedding_module, "wte")
        and hasattr(embedding_module.wte, "weight")
        and isinstance(embedding_module.wte.weight, torch.Tensor)
    ):
        return embedding_module.wte.weight.device

    # Fallback: model parameter device if available
    if model is not None:
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass

    return torch.device("cpu")


@dataclass
class GPTConfig:
    """Configuration class for GPT model parameters"""

    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple
        # of 64 for efficiency
    )
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 0
    n_embd: int = 768
    init_n_embd: int = 0  # Initial embedding dimension (0 means same as
    # n_embd)
    dropout: float = 0.0
    attn_dropout: float = 0.3  # Dropout rate specifically for attention
    residual_scaling_factor: float = 1.0  # Scaling factor for residual
    # connections
    bias: bool = False  # True: bias in Linears and LayerNorms, False: no bias
    residual_scale_init: bool = (
        True  # Controls whether to use special initialization for residual
        # connections
    )
    weight_tying: bool = (
        True  # Controls whether to use weight tying between embedding and
        # LM head
    )
    wte_from_numpy_path: str = ""  # Path to a numpy file containing
    # embeddings for wte
    freeze_wte: bool = False  # Whether to freeze the token embeddings during training
    embeddings_trainable: bool = True  # Whether the vocab matrix remains trainable

    # Embedding adapter config
    use_embedding_adapter: bool = False  # Whether to use embedding adapter
    embedding_adapter_dim: int = 0  # Dimension of embedding adapter input (0 means use n_embd)
    embedding_adapter_hidden_dims: list = None  # List of hidden dimensions for embedding adapter
    embedding_adapter_activation_type: str = "relu"  # Activation function for embedding adapter
    embedding_adapter_dropout: float = 0.0  # Dropout rate for embedding
    # adapter
    use_embedding_adapter_for_logits: bool = True  # Whether to reuse adapter for logits
    logit_chunk_size: int = 0  # Chunk size for logit computation (0 means
    # no chunking)
    use_direct_embeddings: bool = False  # Whether to use direct embeddings
    use_embedding_adapter_bias: bool = True  # Whether adapter linear layers include bias
    flash: bool = False  # Whether to use flash attention
    use_rotary_embeddings: bool = (
        False  # Whether to use rotary embeddings instead of absolute embeddings
    )
    use_relu_squared_mlp: bool = (
        False  # Whether to use ReLU^2 activation in MLP blocks (instead of default)
    )
    use_rms_norm: bool = False  # Apply RMS norm
    mlp_expand_ratio: int = 3  # Expansion ratio for transformer MLP hidden size
    use_mlp_bias: bool = True  # Whether transformer MLP linears include bias

    # NanoChat-style initialization and logit softcap
    use_muparam_init: bool = False  # Use µParam initialization tweaks
    use_logit_softcap: bool = False  # Apply tanh softcap to logits
    logit_softcap_value: float = 15.0  # Softcap value

    # LoRA adapter config (alternative to dense MLP stack)
    use_lora_adapter: bool = False  # Whether to use LoRA adapters instead
    # of dense MLP
    lora_rank: int = 16  # Rank for LoRA decomposition
    lora_alpha: float = 16.0  # Scaling factor for LoRA
    lora_dropout: float = 0.1  # Dropout rate for LoRA layers
    lora_num_layers: int = 4  # Number of LoRA layers to stack
    lora_apply_to_wte: bool = True  # Apply LoRA to word token embeddings
    lora_apply_to_mlp: bool = True  # Apply LoRA to final MLP projection

    # Attention-based embedding adapter config
    use_attention_embedding_adapter: bool = (
        False  # Whether to use attention-based embedding adapter
    )
    attention_adapter_preset: str = (
        "balanced"  # Preset configuration: lightweight, balanced,
        # heavy, custom
    )
    attention_num_layers: int = 2  # Number of attention blocks in
    # attention adapter
    attention_num_heads: int = 8  # Number of attention heads per layer
    attention_ffn_hidden_ratio: float = 4.0  # FFN expansion ratio (hidden_dim = embed_dim * ratio)
    attention_dropout: float = 0.1  # Dropout rate for attention adapter
    attention_use_positional_encoding: bool = True  # Use positional encoding in attention adapter
    attention_use_layer_norm: bool = True  # Use layer normalization in attention adapter
    attention_use_residual: bool = True  # Use residual connections in
    # attention adapter
    attention_activation: str = "gelu"  # Activation function for
    # attention adapter FFN

    # Mixture of Experts config
    n_exp: int = 1  # if n_exp = 1 we just use regular MLP layers
    top_k: int = 2
    use_aux_loss: bool = False  # apply auxiliary loss (from Switch Transformer) in router
    use_router_z_loss: bool = False  # apply router z loss (from ST-MoE)
    use_noisy_top_k: bool = False
    aux_loss_weight: float = (
        0.01  # default setting from Switch Transformer (see top of
        # page 8)
    )
    router_z_loss_weight: float = 0.001  # default setting from ST-MoE (see page 8 eq. 6)
    train_capacity: float = 1.25  # default setting from ST-MoE
    # (see top of page 6)
    eval_capacity: float = 2.0
    min_capacity: int = 4  # minimum batch size to send to any single expert
    stride: int = 2  # one in every stride layers are converted to an MoE
    use_switch_tfm_init: bool = False  # use weight init scheme from
    # Switch Transformer
    switch_tfm_init_scale: float = 1.0
    router_use_full_prec: bool = False  # use float32 precision in the router

    # Event names config
    use_event_names: bool = False
    event_embedding_dim: int = 8
    event_vocab_size: int = 5

    # Label smoothing config
    label_smoothing: float = (
        0.0  # Label smoothing factor (0.0 = no smoothing,
        # 0.1 = 10% smoothing)
    )
    temperature_scaling: float = (
        1.0  # Temperature scaling factor (1.0 = no scaling,
        # >1.0 = softer, <1.0 = sharper)
    )

    # Requested input dtype for model computations (e.g., "float32", "bfloat16")
    input_dtype: Optional[str] = None

    # Relative positional encoding config
    use_relative_position_encoding: bool = (
        False  # Whether to use relative instead of absolute positional
        # encoding
    )
    relative_position_encoding_type: str = (
        "xlnet"  # Type of relative encoding: "xlnet", "transformer_xl",
        # "simple"
    )
    max_relative_position: int = 512  # Maximum relative position
    # distance to encode
    relative_position_clamp: bool = (
        True  # Whether to clamp relative positions to
        # max_relative_position
    )
    use_segment_embeddings: bool = (
        True  # Whether to use segment embeddings (for multi-segment
        # sequences)
    )
    num_segments: int = 2  # Number of segments for segment embeddings
    relative_position_bias: bool = True  # Whether to use bias terms in relative attention
    share_relative_position_bias: bool = (
        False  # Whether to share r_w_bias and r_r_bias across layers
    )

    # XLNet-style bidirectional training config
    use_xlnet_bidirectional_training: bool = (
        False  # Whether to use XLNet-style bidirectional masked modeling
        # for training
    )
    xlnet_permutation_length: int = 6  # Number of tokens to include in
    # each permutation
    xlnet_mask_ratio: float = 0.15  # Ratio of tokens to mask for prediction
    xlnet_target_mapping_type: str = (
        "random"  # Target mapping strategy: "random", "sequential",
        # "last_n"
    )
    xlnet_num_predict_tokens: int = 0  # Number of tokens to predict (0 = use mask_ratio)
    xlnet_two_stream_attention: bool = True  # Use XLNet two-stream
    # attention mechanism
    xlnet_mem_len: int = 0  # Memory length for transformer-xl style
    # memory
    xlnet_reuse_len: int = 0  # Reuse length for memory (0 = no reuse)

    # Merlin-style embedding adapter config
    use_merlin_embedding_adapter: bool = False  # Whether to use Merlin-style embedding processing
    merlin_use_l2_norm: bool = (
        True  # Use L2 normalization instead of LayerNorm in Merlin
        # processing
    )
    activation: str = "relu"  # Activation function for Merlin
    # refinement layers

    debug: bool = False

    def pretty_print(self):
        """
        Return a dictionary representation of the configuration parameters.

        Returns:
            dict: Dictionary containing all configuration parameters for display purposes
        """
        return {
            "block_size": self.block_size,
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "init_n_embd": self.init_n_embd,
            "dropout": self.dropout,
            "attn_dropout": self.attn_dropout,
            "residual_scaling_factor": self.residual_scaling_factor,
            "bias": self.bias,
            "residual_scale_init": self.residual_scale_init,
            "use_rms_norm": self.use_rms_norm,
            "use_muparam_init": self.use_muparam_init,
            "use_logit_softcap": self.use_logit_softcap,
            "logit_softcap_value": self.logit_softcap_value,
            "weight_tying": self.weight_tying,
            "wte_from_numpy_path": self.wte_from_numpy_path,
            "freeze_wte": self.freeze_wte,
            "use_embedding_adapter": self.use_embedding_adapter,
            "embedding_adapter_dim": self.embedding_adapter_dim,
            "embedding_adapter_hidden_dims": (self.embedding_adapter_hidden_dims),
            "embedding_adapter_activation_type": (self.embedding_adapter_activation_type),
            "embedding_adapter_dropout": self.embedding_adapter_dropout,
            "logit_chunk_size": self.logit_chunk_size,
            "use_direct_embeddings": self.use_direct_embeddings,
            "flash": self.flash,
            "use_lora_adapter": self.use_lora_adapter,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_num_layers": self.lora_num_layers,
            "lora_apply_to_wte": self.lora_apply_to_wte,
            "lora_apply_to_mlp": self.lora_apply_to_mlp,
            "use_attention_embedding_adapter": (self.use_attention_embedding_adapter),
            "attention_adapter_preset": self.attention_adapter_preset,
            "attention_num_layers": self.attention_num_layers,
            "attention_num_heads": self.attention_num_heads,
            "attention_ffn_hidden_ratio": self.attention_ffn_hidden_ratio,
            "attention_dropout": self.attention_dropout,
            "attention_use_positional_encoding": (self.attention_use_positional_encoding),
            "attention_use_layer_norm": self.attention_use_layer_norm,
            "attention_use_residual": self.attention_use_residual,
            "attention_activation": self.attention_activation,
            "n_exp": self.n_exp,
            "top_k": self.top_k,
            "use_aux_loss": self.use_aux_loss,
            "use_router_z_loss": self.use_router_z_loss,
            "use_noisy_top_k": self.use_noisy_top_k,
            "aux_loss_weight": self.aux_loss_weight,
            "router_z_loss_weight": self.router_z_loss_weight,
            "train_capacity": self.train_capacity,
            "eval_capacity": self.eval_capacity,
            "min_capacity": self.min_capacity,
            "stride": self.stride,
            "use_switch_tfm_init": self.use_switch_tfm_init,
            "switch_tfm_init_scale": self.switch_tfm_init_scale,
            "router_use_full_prec": self.router_use_full_prec,
            "use_event_names": self.use_event_names,
            "event_embedding_dim": self.event_embedding_dim,
            "event_vocab_size": self.event_vocab_size,
            "label_smoothing": self.label_smoothing,
            "temperature_scaling": self.temperature_scaling,
            "input_dtype": self.input_dtype,
            "use_relative_position_encoding": (self.use_relative_position_encoding),
            "relative_position_encoding_type": (self.relative_position_encoding_type),
            "max_relative_position": self.max_relative_position,
            "relative_position_clamp": self.relative_position_clamp,
            "use_segment_embeddings": self.use_segment_embeddings,
            "num_segments": self.num_segments,
            "relative_position_bias": self.relative_position_bias,
            "share_relative_position_bias": self.share_relative_position_bias,
            "use_xlnet_bidirectional_training": (self.use_xlnet_bidirectional_training),
            "xlnet_permutation_length": self.xlnet_permutation_length,
            "xlnet_mask_ratio": self.xlnet_mask_ratio,
            "xlnet_target_mapping_type": self.xlnet_target_mapping_type,
            "xlnet_num_predict_tokens": self.xlnet_num_predict_tokens,
            "xlnet_two_stream_attention": self.xlnet_two_stream_attention,
            "xlnet_mem_len": self.xlnet_mem_len,
            "xlnet_reuse_len": self.xlnet_reuse_len,
            "use_merlin_embedding_adapter": (self.use_merlin_embedding_adapter),
            "merlin_use_l2_norm": self.merlin_use_l2_norm,
            "activation": self.activation,
            "debug": self.debug
        }


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer that decomposes weight updates into
    two low-rank matrices.

    This implementation provides efficient fine-tuning by adding trainable
    low-rank matrices
    to frozen pre-trained weights, significantly reducing the number of
    trainable parameters
    while maintaining model performance.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.1,
        bias: bool = False,
        init_weights: bool = True,
    ):
        """
        Initialize LoRA layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: Rank of the decomposition (bottleneck dimension)
            alpha: Scaling factor for LoRA output
            dropout: Dropout rate applied to LoRA path
            bias: Whether to include bias term
            init_weights: Whether to initialize LoRA weights
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # Scale factor for LoRA contribution

        # Frozen base linear layer (will be initialized externally if
        # needed)
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)

        # LoRA decomposition: A (in_features x rank) and B (rank x
        # out_features)
        # Forward pass: output = input @ base_weight + (input @ A @ B) *
        # scaling
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout for LoRA path
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize LoRA weights
        if init_weights:
            self.reset_lora_parameters()

    def reset_lora_parameters(self):
        """Initialize LoRA parameters using Kaiming uniform for A and
        zeros for B."""
        # Initialize A with small random values (similar to Kaiming
        # uniform)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initialize B with zeros (standard LoRA practice)
        nn.init.zeros_(self.lora_B)

    def freeze_base_layer(self):
        """Freeze the base layer parameters."""
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def unfreeze_base_layer(self):
        """Unfreeze the base layer parameters."""
        for param in self.base_layer.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Forward pass combining base layer and LoRA adaptation.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Base transformation
        base_output = self.base_layer(x)

        # LoRA transformation: x @ A @ B
        lora_output = x @ self.lora_A.T  # (..., in_features) @ (in_features, rank) -> (..., rank)
        lora_output = self.dropout(lora_output)
        lora_output = lora_output @ self.lora_B.T  # (..., rank) @ (rank, out_features) -> (...,
        # out_features)

        # Combine base and LoRA outputs
        return base_output + lora_output * self.scaling

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}"
        )


class LoRAEmbeddingAdapter(nn.Module):
    """
    LoRA-based Embedding Adapter that replaces dense MLP stack with
    efficient LoRA layers.

    This adapter applies LoRA transformations to both token embeddings
    and stacked projections, providing faster convergence compared to dense
    projections while using fewer parameters.
    """

    def __init__(
        self,
        config,
        wte=None,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.1,
        num_layers: int = 4,
        apply_to_wte: bool = True,
        apply_to_mlp: bool = True,
    ):
        """
        Initialize LoRA embedding adapter.

        Args:
            config: GPTConfig containing model configuration
            wte: Optional pre-existing word token embedding layer
            rank: Rank for LoRA decomposition
            alpha: Scaling factor for LoRA
            dropout: Dropout rate for LoRA layers
            num_layers: Number of LoRA layers to stack
            apply_to_wte: Whether to apply LoRA to word token embeddings
            apply_to_mlp: Whether to apply LoRA to MLP projections
        """
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        self.rank = rank
        self.alpha = alpha
        self.num_layers = num_layers
        self.apply_to_wte = apply_to_wte
        self.apply_to_mlp = apply_to_mlp

        # Determine embedding dimension
        self.wte_dim = config.n_embd if config.init_n_embd == 0 else config.init_n_embd
        self.use_direct_embeddings = getattr(config, "use_direct_embeddings", False)

        # Token embeddings with optional LoRA
        if wte is not None:
            self.wte = wte
        elif not self.use_direct_embeddings:
            self.wte = nn.Embedding(config.vocab_size, self.wte_dim)
        else:
            self.wte = None

        # LoRA transformation for token embeddings (if enabled)
        if self.apply_to_wte and not self.use_direct_embeddings:
            self.wte_lora = LoRALayer(
                in_features=self.wte_dim,
                out_features=self.wte_dim,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                bias=False,
            )
        else:
            self.wte_lora = None

        # Stack of LoRA layers for MLP projections (if enabled)
        if self.apply_to_mlp and num_layers > 0:
            self.lora_layers = nn.ModuleList()

            for i in range(num_layers):
                # All intermediate layers have the same dimensions
                self.lora_layers.append(
                    LoRALayer(
                        in_features=self.wte_dim,
                        out_features=self.wte_dim,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                        bias=False,
                    )
                )

            # Final projection to target embedding dimension if needed
            if self.wte_dim != self.n_embd:
                self.final_projection = LoRALayer(
                    in_features=self.wte_dim,
                    out_features=self.n_embd,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    bias=False,
                )
            else:
                self.final_projection = None
        else:
            self.lora_layers = nn.ModuleList()
            self.final_projection = None

        # Layer normalization for stability
        self.ln_input = LayerNorm(self.wte_dim, bias=config.bias)
        self.ln_output = LayerNorm(self.n_embd, bias=config.bias)

        # L2 normalization
        self.use_l2_norm = True
        self.norm = RMSNorm() if self.use_rms_norm else L2Norm(dim=-1, eps=1e-12)
        # self.l2_norm = L2Norm(dim=-1, eps=1e-12)

        # Store chunk size for logit projection
        self.logit_chunk_size = config.logit_chunk_size

    def forward(self, idx):
        """
        Forward pass through LoRA embedding adapter.

        Args:
            idx: Token indices [batch_size, seq_len] or embeddings
                [batch_size, seq_len, dim]

        Returns:
            Adapted embeddings [batch_size, seq_len, n_embd]
        """
        # Handle direct embeddings or token lookup
        if self.use_direct_embeddings:
            tok_emb = idx  # Input is already embeddings
        else:
            tok_emb = self.wte(idx)  # Lookup token embeddings

        # Apply input layer normalization
        x = self.ln_input(tok_emb)

        # Apply LoRA to token embeddings if enabled
        if self.wte_lora is not None:
            x = self.wte_lora(x)

        # Apply stacked LoRA layers if enabled
        if self.apply_to_mlp:
            for lora_layer in self.lora_layers:
                # Residual connection through each LoRA layer
                x = x + lora_layer(x)

            # Final projection if dimension mismatch
            if self.final_projection is not None:
                x = self.final_projection(x)

        # Apply L2 normalization if enabled
        # if self.use_l2_norm:
        #     x = self.l2_norm(x)

        x = self.norm(x)

        # Apply output layer normalization
        x = self.ln_output(x)

        return x

    def freeze_base_weights(self):
        """Freeze all base layer weights, keeping only LoRA parameters
        trainable."""
        if self.wte is not None:
            for param in self.wte.parameters():
                param.requires_grad = False

        if self.wte_lora is not None:
            self.wte_lora.freeze_base_layer()

        for lora_layer in self.lora_layers:
            lora_layer.freeze_base_layer()

        if self.final_projection is not None:
            self.final_projection.freeze_base_layer()

    def unfreeze_base_weights(self):
        """Unfreeze all base layer weights for full fine-tuning."""
        if self.wte is not None:
            for param in self.wte.parameters():
                param.requires_grad = True

        if self.wte_lora is not None:
            self.wte_lora.unfreeze_base_layer()

        for lora_layer in self.lora_layers:
            lora_layer.unfreeze_base_layer()

        if self.final_projection is not None:
            self.final_projection.unfreeze_base_layer()

    def get_lora_parameters(self):
        """Get only the LoRA parameters for efficient optimization."""
        lora_params = []

        if self.wte_lora is not None:
            lora_params.extend([self.wte_lora.lora_A, self.wte_lora.lora_B])

        for lora_layer in self.lora_layers:
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])

        if self.final_projection is not None:
            lora_params.extend([self.final_projection.lora_A, self.final_projection.lora_B])

        return lora_params

    def get_num_lora_parameters(self):
        """Get the number of trainable LoRA parameters."""
        return sum(p.numel() for p in self.get_lora_parameters())

    def clip_gradients(self, max_norm):
        """Clip gradients of LoRA parameters."""
        if max_norm > 0:
            return nn.utils.clip_grad_norm_(self.get_lora_parameters(), max_norm)
        return 0.0


class EmbeddingAdapter(nn.Module):
    """
    Adapter layer for token embeddings that handles the embedding lookup
    and applies transformations.
    """

    def __init__(
        self,
        config,
        wte=None,
        hidden_dims=None,
        activations=None,
        no_activation_last_layer=True,
        dropout=None,
        without_hidden_layers=False,
    ):
        super().__init__()
        self.n_embd = config.n_embd
        self.vocab_size = config.vocab_size
        self.without_hidden_layers = without_hidden_layers
        # Determine embedding dimension
        self.wte_dim = config.n_embd if config.init_n_embd == 0 else config.init_n_embd
        # Store direct embeddings flag
        self.use_direct_embeddings = getattr(config, "use_direct_embeddings", False)

        # Use externally created token embedding layer
        if wte is not None:
            self.wte = wte
        else:
            self.wte = (
                None
                if self.use_direct_embeddings
                else nn.Embedding(config.vocab_size, self.wte_dim)
            )

        # Set dropout rate
        dropout_rate = dropout if dropout is not None else config.embedding_adapter_dropout

        # If hidden_dims not provided, use config's embedding_adapter_dim
        if hidden_dims is None:
            # If embedding_adapter_dim is 0, use the same size as n_embd
            default_hidden_dim = (
                config.embedding_adapter_dim if config.embedding_adapter_dim > 0 else config.n_embd
            )
            hidden_dims = [default_hidden_dim]

        if without_hidden_layers:
            hidden_dims = []

        # Setup activations (default to ReLU)
        if activations is None:
            activations = [nn.ReLU() for _ in range(len(hidden_dims))]

        # Remove activation from last layer if specified
        # if no_activation_last_layer and len(activations) > 0:
        #     activations[-1] = nn.Identity()

        # Create projection layers
        layers = []
        # Use the actual embedding dimension as input dimension
        input_dim = self.wte_dim

        adapter_bias = getattr(config, "use_embedding_adapter_bias", True)

        # Build MLP layers
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, dim, bias=adapter_bias))

            if i != len(hidden_dims) - 1:
                layers.append(activations[i])
            else:
                if not no_activation_last_layer:
                    layers.append(activations[i])

            if dropout_rate is not None and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            input_dim = dim

        # Final projection back to embedding dimension if needed
        if input_dim != self.n_embd:
            layers.append(nn.Linear(input_dim, self.n_embd, bias=adapter_bias))

        # Combine all layers into a sequential module
        if len(layers) > 0:
            self.adapter_layers = nn.Sequential(*layers)
        else:
            self.adapter_layers = nn.Identity()

        # Layer norm before and after adapter - helps with gradient stability
        # self.ln_pre = LayerNorm(self.wte_dim, bias=config.bias)
        # self.ln_post = LayerNorm(self.n_embd, bias=config.bias)

        # Store configuration
        self.no_activation_last_layer = no_activation_last_layer
        self.use_rms_norm = getattr(
            config,
            "use_rms_norm",
            getattr(config, "use_rms_norm", False),
        )

        # Create L2 normalization layer
        self.use_l2_norm = True

        # TODO COMMENT ME
        self.norm = L2Norm(
            dim=-1, eps=1e-12
        )  # RMSNorm() if self.use_rms_norm else L2Norm(dim=-1, eps=1e-12)

        # Store the chunk size for logit projection
        self.logit_chunk_size = config.logit_chunk_size

    @staticmethod
    def copy_adapter(source_adapter):
        """Create a copy of the given EmbeddingAdapter with the same adapter layers."""
        if not isinstance(source_adapter, EmbeddingAdapter):
            raise TypeError("source_adapter must be an EmbeddingAdapter instance.")

        config = source_adapter.config
        new_adapter = EmbeddingAdapter(
            config=config,
            wte=source_adapter.wte,
            hidden_dims=None,
            activations=None,
            no_activation_last_layer=source_adapter.no_activation_last_layer,
            dropout=source_adapter.logit_chunk_size,
            without_hidden_layers=source_adapter.without_hidden_layers,
        )
        new_adapter.adapter_layers.load_state_dict(source_adapter.adapter_layers.state_dict())
        return new_adapter

    def adapt(self, x, training=True, for_logits=False):
        """Apply the adapter transformation to embeddings

        Args:
            x: Input embeddings
            training: Whether to compute gradients for this operation
        """
        # Apply layer norm before adapter (if defined)
        if hasattr(self, "ln_pre") and self.ln_pre is not None:
            norm_x = self.ln_pre(x)
        else:
            norm_x = x

        if not training:
            with torch.no_grad():
                # Apply adapter layers
                h = self.adapter_layers(norm_x)
        else:
            # Apply adapter layers
            h = self.adapter_layers(norm_x)

        # Residual connection
        # out = x + h
        # out = h

        # Apply L2 normalization if enabled
        # if self.use_rms_norm:
        #     h = F.rms_norm(h, (h.size(-1),))
        # elif self.use_l2_norm:
        #     h = self.norm(h)

        # TODO COMMENT ME
        # if not for_logits:
        #     h = self.norm(h)
        h = self.norm(h)

        # Final layer norm (if defined)
        if hasattr(self, "ln_post") and self.ln_post is not None:
            return self.ln_post(h)
        else:
            return h

    def forward(self, idx, training=True):
        """
        Lookup embeddings for input tokens and apply adapter

        Args:
            idx: token indices of shape [batch_size, seq_len]
                 or embeddings of shape [batch_size, seq_len, embedding_dim]
                 when use_direct_embeddings=True

        Returns:
            Adapted token embeddings of shape [batch_size, seq_len, n_embd]
        """
        # Handle direct embeddings if enabled

        # print('idx dtype', idx.dtype)
        if self.use_direct_embeddings:
            # Input is already embeddings, no need for lookup
            tok_emb = idx  # [batch_size, seq_len, embedding_dim]
        else:
            # Lookup token embeddings
            tok_emb = self.wte(idx)  # [batch_size, seq_len, n_embd]

        if self.without_hidden_layers:
            return tok_emb

        # do it if tok_emb size  > 2
        # size = tok_emb.size()
        # if len(size) > 2:
        #     b, t, n_embd = size
        #     # print("tok_emb", tok_emb.shape)
        #     # reshape tok_emb to [-1, n_emb]
        #     tok_emb = tok_emb.reshape(-1, n_embd)

        adapted_emb = self.adapt(tok_emb, training=training)
        # print('adapted_emb dtype', adapted_emb.dtype)

        # reshape adapted_emb to [b, t, n_embd]
        # if len(size) > 2:
        #     adapted_emb = adapted_emb.reshape(b, t, -1)
        return adapted_emb

    def clip_gradients(self, max_norm):
        """
        Clip gradients of the adapter's parameters to prevent explosion.

        Args:
            max_norm: Maximum norm for the gradients.
        """
        if max_norm > 0:
            return nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        return 0.0


class CustomEmbeddings(object):
    def __init__(self, custom_embeddings, embedding_adapter):
        self.custom_embeddings = custom_embeddings
        self.embedding_adapter = embedding_adapter

    @property
    def embeddings(self):
        return self.embedding_adapter(
            self.custom_embeddings.weight.to(
                device=self.embedding_adapter.adapter_layers[0].weight.device,
                # dtype=hidden_states.dtype
            ),
            training=True,
        )


class PredictionHead(nn.Module):
    """
    Prediction head for language modeling.
    Handles embedding adapter or direct logits computation.
    """

    def __init__(
        self,
        embedding_adapter,
        custom_embeddings=None,
        chunk_size=0,
        wte_from_numpy_path=None,
        verbose=False,
        trainable_embeddings=False,
        use_adapter_for_logits=True,
        dtype="float32"
    ):
        """
        Initialize prediction head.

        Args:
            embedding_adapter: The embedding adapter to use
            chunk_size: Size of chunks for processing (0 means no chunking)
            wte_from_numpy_path: Path to numpy file with embeddings
        """
        super().__init__()
        self.embedding_adapter = embedding_adapter
        self.chunk_size = chunk_size
        self.wte_from_numpy_path = wte_from_numpy_path
        self.custom_embeddings = custom_embeddings
        self.verbose = verbose
        self.trainable_embeddings = trainable_embeddings
        self.use_adapter_for_logits = use_adapter_for_logits
        self.dtype = dtype
        # Check if direct embeddings are enabled
        self.use_direct_embeddings = getattr(self.embedding_adapter, "use_direct_embeddings", False)

        # If direct embeddings are provided through wte_from_numpy_path, load
        # them
        if self.use_direct_embeddings and self.wte_from_numpy_path:
            # return
            try:
                import numpy as np

                if self.verbose:
                    print(f"PredictionHead: Loading embeddings from {self.wte_from_numpy_path}")
                embeddings = np.load(self.wte_from_numpy_path)

                # Ensure the embeddings have the right shape
                emb_shape = embeddings.shape

                self.custom_embeddings = nn.Embedding(emb_shape[0], emb_shape[1])
                self.custom_embeddings.weight.data.copy_(torch.from_numpy(embeddings))
                self.custom_embeddings.weight.requires_grad = self.trainable_embeddings
                # if self.dtype == "bfloat16":
                #     self.custom_embeddings = self.custom_embeddings.to(dtype=torch.bfloat16)

                # if self.trainable_embeddings:
                #     self.custom_embeddings = nn.Embedding(emb_shape[0], emb_shape[1])
                #     self.custom_embeddings.weight.data.copy_(torch.from_numpy(embeddings))
                #     self.custom_embeddings.weight.requires_grad = True
                # else:
                #     buffer_tensor = torch.as_tensor(embeddings, dtype=torch.float32)
                #     self.register_buffer(
                #         "_custom_embeddings_buffer", buffer_tensor, persistent=False
                #     )

                print(f"PredictionHead: Successfully loaded embeddings with shape {emb_shape}")
            except (OSError, ValueError, RuntimeError) as e:
                print(
                    f"PredictionHead: Error loading embeddings from {self.wte_from_numpy_path}: {e}"
                )
                self.custom_embeddings = None
                if hasattr(self, "_custom_embeddings_buffer"):
                    delattr(self, "_custom_embeddings_buffer")

        else:
            # Get vocabulary size from embedding layer
            if hasattr(self.embedding_adapter, "wte") and self.embedding_adapter.wte is not None:
                self.vocab_size = self.embedding_adapter.wte.weight.shape[0]
            else:
                # Fallback to config vocab_size when using direct embeddings
                self.vocab_size = getattr(self.embedding_adapter, "vocab_size", 50304)
            self.custom_embeddings = None
            if self.verbose:
                print(f"PredictionHead: Vocab size: {self.vocab_size}")

    def forward(self, hidden_states, training=True):
        """
        Project hidden states to logits using adapted token embeddings

        CRITICAL: When embeddings_trainable=False, we freeze the BASE embeddings
        but the ADAPTER MLP must remain trainable for weight tying to work!
        """
        import torch

        if torch.isnan(hidden_states).any():
            print("WARNING: NaN detected in hidden_states input to prediction head")

        if not self.use_adapter_for_logits:
            return self._project_logits_without_adapter(hidden_states)

        # Process all tokens at once if chunk_size is 0
        if self.chunk_size == 0:
            try:
                # Get vocabulary embeddings
                if self.custom_embeddings is not None:
                    # Custom embeddings from numpy
                    vocab_embeddings = self.custom_embeddings.weight
                elif self.use_direct_embeddings:
                    # Direct embeddings mode
                    device = hidden_states.device
                    dtype = hidden_states.dtype
                    emb_dim = getattr(self.embedding_adapter, "wte_dim", hidden_states.size(-1))
                    vocab_embeddings = torch.randn(
                        self.vocab_size, emb_dim, device=device, dtype=dtype
                    )
                else:
                    # Standard token indices
                    all_indices = torch.arange(0, self.vocab_size, device=hidden_states.device)
                    # Get base embeddings (these might be frozen)
                    vocab_embeddings = self.embedding_adapter.wte(all_indices)

                # CRITICAL FIX: Always allow gradients through the adapter MLP!
                # Only the base embeddings (wte) are frozen, not the adapter layers
                # if hasattr(vocab_embeddings, "requires_grad"):
                #     vocab_embeddings = vocab_embeddings.detach()  # Detach frozen embeddings

                if not training and self.dtype == "bfloat16":
                    vocab_embeddings = vocab_embeddings.to(dtype=torch.float32)


                # Now apply the adapter transformation WITH gradients
                adapted_emb = self.embedding_adapter.adapt(
                    vocab_embeddings, training=training, for_logits=True
                )

                # Ensure compatibility
                adapted_emb = adapted_emb.to(device=hidden_states.device, dtype=hidden_states.dtype)

                # Compute logits
                logits = torch.matmul(hidden_states, adapted_emb.t())

                if torch.isnan(logits).any():
                    print("WARNING: NaN detected in logits after matrix multiplication")

                return logits

            except (RuntimeError, ValueError) as e:
                print(f"Error in full vocab logit projection: {e}")
                raise

    # def forward(self, hidden_states):
    #     """
    #     Project hidden states to logits using adapted token embeddings
    #
    #     Args:
    #         hidden_states: model hidden states of shape [batch_size, seq_len, n_embd]
    #
    #     Returns:
    #         Logits of shape [batch_size, seq_len, vocab_size]
    #     """
    #     # Ensure torch is accessible (fix for UnboundLocalError)
    #     import torch
    #
    #     # Check for NaN in input
    #     if torch.isnan(hidden_states).any():
    #         print("WARNING: NaN detected in hidden_states input to prediction head")
    #
    #     # print("hidden_states shape", hidden_states.shape)
    #     # print("hidden_states device", hidden_states.device)
    #     # print("hidden_states dtype", hidden_states.dtype)
    #
    #     # Check if using LoRA adapter and set up dynamo suppression if needed
    #     is_lora_adapter = hasattr(self.embedding_adapter, "lora_layers")
    #     original_suppress = None
    #
    #     if is_lora_adapter:
    #         # Suppress torch._dynamo errors for LoRA compatibility
    #         try:
    #             import torch._dynamo
    #
    #             original_suppress = torch._dynamo.config.suppress_errors
    #             torch._dynamo.config.suppress_errors = True
    #         except (ImportError, AttributeError):
    #             original_suppress = None
    #
    #     try:
    #         if not self.use_adapter_for_logits:
    #             return self._project_logits_without_adapter(hidden_states)
    #
    #         # Process all tokens at once if chunk_size is 0
    #         if self.chunk_size == 0:
    #             try:
    #                 # Priority 1: Use custom embeddings from numpy file if
    #                 # available
    #                 custom_tensor = self._get_custom_embeddings_tensor()
    #                 if custom_tensor is not None:
    #                     # Use the custom embeddings directly for projection
    #                     # Wrap LoRA operations in no_grad for stability
    #                     if is_lora_adapter and not self.trainable_embeddings:
    #                         with torch.no_grad():
    #                             adapted_emb = self.embedding_adapter(custom_tensor)
    #                     else:
    #                         adapted_emb = self.embedding_adapter(custom_tensor)
    #
    #                 # if True:
    #                 #     adapted_emb = self.custom_embeddings.embeddings
    #
    #                 else:
    #                     # Handle direct embeddings mode differently
    #                     if self.use_direct_embeddings:
    #                         # For direct embeddings, create random embeddings
    #                         # for all vocab
    #                         device = hidden_states.device
    #                         dtype = hidden_states.dtype
    #                         # Get embedding dimension from the adapter
    #                         emb_dim = getattr(
    #                             self.embedding_adapter,
    #                             "wte_dim",
    #                             hidden_states.size(-1),
    #                         )
    #
    #                         # Create random embeddings matrix for all
    #                         # vocabulary
    #                         all_embeddings = torch.randn(
    #                             self.vocab_size, emb_dim, device=device, dtype=dtype
    #                         )
    #
    #                         # Get adapted embeddings for all vocabulary tokens
    #                         if is_lora_adapter:
    #                             with torch.no_grad():
    #                                 adapted_emb = self.embedding_adapter(all_embeddings)
    #                         else:
    #                             adapted_emb = self.embedding_adapter(all_embeddings)
    #                     else:
    #                         # Standard approach: get all token indices and
    #                         # process through the adapter
    #                         all_indices = torch.arange(
    #                             0, self.vocab_size, device=hidden_states.device
    #                         )
    #
    #                         # Get adapted embeddings for all vocabulary tokens
    #                         if is_lora_adapter:
    #                             with torch.no_grad():
    #                                 adapted_emb = self.embedding_adapter(all_indices)
    #                         else:
    #                             adapted_emb = self.embedding_adapter(all_indices)
    #
    #                 # Ensure tensor compatibility and handle potential memory
    #                 # issues
    #                 adapted_emb = adapted_emb.to(
    #                     device=hidden_states.device, dtype=hidden_states.dtype
    #                 )
    #
    #                 # Check for NaN in adapted embeddings
    #                 if torch.isnan(adapted_emb).any():
    #                     print("WARNING: NaN detected in adapted embeddings")
    #                     # Replace NaN with small random values
    #                     adapted_emb = torch.where(
    #                         torch.isnan(adapted_emb),
    #                         torch.randn_like(adapted_emb) * 0.02,
    #                         adapted_emb,
    #                     )
    #
    #                 # Compute logits using matrix multiplication with complete
    #                 # CUDA recovery
    #                 logits = torch.matmul(hidden_states, adapted_emb.t())
    #
    #                 # Check for NaN in logits
    #                 if torch.isnan(logits).any():
    #                     print("WARNING: NaN detected in logits after matrix multiplication")
    #
    #                 return logits
    #             except (RuntimeError, ValueError) as e:
    #                 print(f"Error in full vocab logit projection: {e}")
    #                 raise
    #         else:
    #             # Process in chunks to save memory
    #             batch_size, seq_len, _ = hidden_states.size()
    #             logits = torch.zeros(
    #                 (batch_size, seq_len, self.vocab_size),
    #                 dtype=hidden_states.dtype,
    #                 device=hidden_states.device,
    #             )
    #
    #             # Process vocabulary in chunks
    #             for i in range(0, self.vocab_size, self.chunk_size):
    #                 end_idx = min(i + self.chunk_size, self.vocab_size)
    #
    #                 # Priority 1: Use custom embeddings if available
    #                 custom_tensor = self._get_custom_embeddings_tensor()
    #                 if custom_tensor is not None:
    #                     # Get embeddings directly from custom embeddings
    #                     chunk_emb = custom_tensor[i:end_idx]
    #
    #                 # Priority 2: Handle direct embeddings mode
    #                 elif self.use_direct_embeddings:
    #                     # Get indices for this chunk
    #                     chunk_indices = torch.arange(i, end_idx, device=hidden_states.device)
    #                     # Get embeddings through adapter (which should handle
    #                     # direct embeddings mode)
    #                     if is_lora_adapter:
    #                         with torch.no_grad():
    #                             chunk_emb = self.embedding_adapter(chunk_indices)
    #                     else:
    #                         chunk_emb = self.embedding_adapter(chunk_indices)
    #
    #                 # Priority 3: Standard approach
    #                 else:
    #                     # Get indices for this chunk
    #                     chunk_indices = torch.arange(i, end_idx, device=hidden_states.device)
    #                     # Get adapted embeddings for chunk using forward
    #                     if is_lora_adapter:
    #                         with torch.no_grad():
    #                             chunk_emb = self.embedding_adapter(chunk_indices)
    #                     else:
    #                         chunk_emb = self.embedding_adapter(chunk_indices)
    #
    #                 # Check for NaN in chunk embeddings
    #                 if torch.isnan(chunk_emb).any():
    #                     print(f"WARNING: NaN detected in chunk embeddings {i}:{end_idx}")
    #
    #                 # Compute chunk logits using matrix multiplication
    #                 chunk_logits = torch.matmul(hidden_states, chunk_emb.t())
    #
    #                 # Store chunk logits in the appropriate location
    #                 logits[:, :, i:end_idx] = chunk_logits
    #
    #             # Check for NaN in final logits
    #             if torch.isnan(logits).any():
    #                 print("WARNING: NaN detected in final chunked logits")
    #
    #             return logits
    #
    #     finally:
    #         # Restore original torch._dynamo suppress_errors setting
    #         if is_lora_adapter and original_suppress is not None:
    #             try:
    #                 import torch._dynamo
    #
    #                 torch._dynamo.config.suppress_errors = original_suppress
    #             except (ImportError, AttributeError):
    #                 pass

    def _project_logits_without_adapter(self, hidden_states):
        """Project logits using the raw vocab matrix without running the adapter."""
        weight = self._get_vocab_weight()
        adapted_weight = self._adapt_vocab_embeddings(weight)
        return self._multiply_hidden_states(hidden_states, adapted_weight)

    def _get_custom_embeddings_tensor(self):
        """Return the custom embeddings tensor (parameter or buffer)."""
        if self.custom_embeddings is not None:
            return self.custom_embeddings.weight
        return getattr(self, "_custom_embeddings_buffer", None)

    def _get_vocab_weight(self):
        """Return the vocabulary matrix used for logits."""
        if self.custom_embeddings is not None:
            return self.custom_embeddings.weight
        buffer_tensor = getattr(self, "_custom_embeddings_buffer", None)
        if buffer_tensor is not None:
            return buffer_tensor
        if hasattr(self.embedding_adapter, "wte") and self.embedding_adapter.wte is not None:
            return self.embedding_adapter.wte.weight
        raise RuntimeError(
            "PredictionHead requires either custom embeddings or an embedding layer to project logits."
        )

    def _adapt_vocab_embeddings(self, embeddings):
        """Project raw vocab embeddings through the adapter if needed."""
        is_lora_adapter = hasattr(self.embedding_adapter, "lora_layers")
        if is_lora_adapter and not self.trainable_embeddings:
            with torch.no_grad():
                return self.embedding_adapter(embeddings)
        return self.embedding_adapter(embeddings)

    def _multiply_hidden_states(self, hidden_states, weight):
        """Multiply hidden states by the vocab matrix."""
        import torch

        weight = weight.to(device=hidden_states.device, dtype=hidden_states.dtype)
        logits = torch.matmul(hidden_states, weight.t())
        if torch.isnan(logits).any():
            print("WARNING: NaN detected in logits when using raw vocab matrix")
        return logits


def label_smoothing_cross_entropy(
    logits, targets, label_smoothing=0.1, temperature_scaling=1.0, ignore_index=-1
):
    """
    Compute cross-entropy loss with label smoothing and temperature scaling.

    Args:
        logits: Model predictions of shape [batch_size, vocab_size] or [batch_size * seq_len, vocab_size]
        targets: Target labels of shape [batch_size] or [batch_size * seq_len]
        label_smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
        temperature_scaling: Temperature scaling factor (1.0 = no scaling, >1.0 = softer, <1.0 = sharper)
        ignore_index: Index to ignore in loss calculation

    Returns:
        Smoothed cross-entropy loss with temperature scaling
    """
    # Apply temperature scaling to logits
    if temperature_scaling != 1.0:
        logits = logits / temperature_scaling

    if label_smoothing == 0.0:
        # No label smoothing, use standard cross-entropy
        return F.cross_entropy(logits, targets, ignore_index=ignore_index)

    # Get vocabulary size
    vocab_size = logits.size(-1)

    # Create mask for valid targets (not ignore_index)
    valid_mask = targets != ignore_index

    if not valid_mask.any():
        # No valid targets, return zero loss
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Filter out ignored indices
    valid_logits = logits[valid_mask]
    valid_targets = targets[valid_mask]

    # Compute log probabilities
    log_probs = F.log_softmax(valid_logits, dim=-1)

    # Create smoothed target distribution
    # (1 - label_smoothing) for true class, label_smoothing / (vocab_size - 1) for others
    smooth_targets = torch.full_like(log_probs, label_smoothing / (vocab_size - 1))
    smooth_targets.scatter_(1, valid_targets.unsqueeze(1), 1.0 - label_smoothing)

    # Compute loss as negative log-likelihood with smoothed targets
    loss = -(smooth_targets * log_probs).sum(dim=-1).mean()

    return loss


class GPT(nn.Module):
    """
    GPT transformer model for language modeling.

    A flexible GPT implementation supporting various architectural configurations
    including LoRA adapters, embedding adapters, mixture of experts, and
    relative positional encoding. The model can be configured for both training
    and inference modes with optional verbosity for debugging.

    Args:
        config (GPTConfig): Configuration object containing model hyperparameters
        verbose (bool): Whether to print detailed model information during initialization
        inference (bool): Whether to configure the model for inference mode
    """

    def __init__(self, config, verbose=False, inference=False):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config
        self.verbose = verbose
        self.inference = inference
        self.use_rms_norm = getattr(config, "use_rms_norm", False)
        self.embedding_rms_norm = None  # RMSNorm() if self.use_rms_norm else None
        self.use_event_names = getattr(config, "use_event_names", False)
        self.event_embedding_dim = getattr(config, "event_embedding_dim", 8)
        self.event_vocab_size = getattr(config, "event_vocab_size", 5)

        if getattr(config, "use_rotary_embeddings", False) and getattr(
            config, "use_relative_position_encoding", False
        ):
            raise ValueError(
                "Rotary embeddings cannot be combined with relative positional encoding."
            )

        # Setup components
        shared_relative_biases = self._setup_relative_position_encoding(config)
        blocks = self._create_transformer_blocks(config, shared_relative_biases)
        transformer_wte = self._create_embedding_layer(config)

        # Create transformer components
        self.transformer = self._create_transformer_components(config, transformer_wte, blocks)

        # Initialize shared rotary embeddings if enabled
        if getattr(config, "use_rotary_embeddings", False):
            self._initialize_rotary_cache(config, transformer_wte)
            self._propagate_rotary_cache_to_blocks()

        # Setup event embeddings if needed
        self._setup_event_embeddings(config)

        # Create language model head
        self._create_language_model_head(config, transformer_wte)

        # Initialize weights
        self._initialize_weights(config)

        # Load and configure embeddings
        self._setup_embeddings(config)

        # Print model information if verbose
        if self.verbose:
            self._print_model_summary(config)

    def _setup_relative_position_encoding(self, config):
        """Setup relative positional encoding and return shared bias terms if needed."""
        shared_relative_biases = None
        if getattr(config, "use_relative_position_encoding", False):
            if self.verbose:
                print("Using relative positional encoding")
                print(f"- Type: {config.relative_position_encoding_type}")
                print(f"- Max relative position: {config.max_relative_position}")
                print(f"- Use segment embeddings: {config.use_segment_embeddings}")
                print(f"- Share bias terms: {config.share_relative_position_bias}")

            # Create shared bias terms if configured
            if config.share_relative_position_bias and config.relative_position_bias:
                head_dim = config.n_embd // config.n_head
                shared_relative_biases = {
                    "r_w_bias": nn.Parameter(torch.zeros(config.n_head, head_dim)),
                    "r_r_bias": nn.Parameter(torch.zeros(config.n_head, head_dim)),
                }
                # Register as parameters of the main model
                for name, param in shared_relative_biases.items():
                    self.register_parameter(f"shared_{name}", param)
        return shared_relative_biases

    def _create_embedding_layer(self, config):
        """Create the appropriate embedding layer based on configuration."""
        actual_emb_dim = config.n_embd if config.init_n_embd == 0 else config.init_n_embd

        if config.use_embedding_adapter:
            return self._create_embedding_adapter(config, actual_emb_dim)
        else:
            return nn.Embedding(config.vocab_size, actual_emb_dim)

    def _create_embedding_adapter(self, config, actual_emb_dim):
        """Create embedding adapter based on configuration type."""
        if self.verbose:
            print("Using embedding adapter")

        base_embedding = (
            None
            if config.use_direct_embeddings
            else nn.Embedding(config.vocab_size, actual_emb_dim)
        )

        if config.use_lora_adapter:
            return self._create_lora_adapter(config, base_embedding)
        elif config.use_attention_embedding_adapter:
            return self._create_attention_adapter(config, base_embedding)
        else:
            return self._create_dense_adapter(config, base_embedding)

    def _create_lora_adapter(self, config, base_embedding):
        """Create LoRA-based embedding adapter."""
        if self.verbose:
            print("Using LoRA-based embedding adapter")
            print(f"- LoRA rank: {config.lora_rank}")
            print(f"- LoRA alpha: {config.lora_alpha}")
            print(f"- LoRA layers: {config.lora_num_layers}")
            print(f"- Apply to WTE: {config.lora_apply_to_wte}")
            print(f"- Apply to MLP: {config.lora_apply_to_mlp}")

        return LoRAEmbeddingAdapter(
            config,
            wte=base_embedding,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
            num_layers=config.lora_num_layers,
            apply_to_wte=config.lora_apply_to_wte,
            apply_to_mlp=config.lora_apply_to_mlp,
        )

    def _create_attention_adapter(self, config, base_embedding):
        """Create attention-based embedding adapter."""
        from .attention_embedding_adapter import create_attention_embedding_adapter

        if self.verbose:
            print("Using attention-based embedding adapter")
            print(f"- Preset: {config.attention_adapter_preset}")
            print(f"- Attention layers: {config.attention_num_layers}")
            print(f"- Attention heads: {config.attention_num_heads}")
            print(f"- FFN ratio: {config.attention_ffn_hidden_ratio}")
            print(f"- Dropout: {config.attention_dropout}")
            print(f"- Positional encoding: {config.attention_use_positional_encoding}")
            print(f"- Layer norm: {config.attention_use_layer_norm}")
            print(f"- Residual connections: {config.attention_use_residual}")
            print(f"- Activation: {config.attention_activation}")

        return create_attention_embedding_adapter(
            config,
            wte=base_embedding,
            preset=config.attention_adapter_preset,
            num_layers=config.attention_num_layers,
            num_heads=config.attention_num_heads,
            ffn_hidden_ratio=config.attention_ffn_hidden_ratio,
            dropout=config.attention_dropout,
            use_positional_encoding=config.attention_use_positional_encoding,
            use_layer_norm=config.attention_use_layer_norm,
            use_residual=config.attention_use_residual,
            activation=config.attention_activation,
        )

    def _create_dense_adapter(self, config, base_embedding):
        """Create dense MLP embedding adapter."""
        if self.verbose:
            print("Using dense MLP embedding adapter")

        hidden_dims = config.embedding_adapter_hidden_dims
        activations = self._create_activations(config, hidden_dims) if hidden_dims else None

        return EmbeddingAdapter(
            config,
            wte=base_embedding,
            hidden_dims=hidden_dims,
            activations=activations,
            without_hidden_layers=False,
        )

    def _create_activations(self, config, hidden_dims):
        """Create activation functions based on configuration."""
        if not hidden_dims:
            return None

        activation_type = config.embedding_adapter_activation_type.lower()
        activation_map = {
            "relu": nn.ReLU,
            "gelu": NewGELU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
        }

        activation_class = activation_map.get(activation_type, nn.ReLU)
        return [activation_class() for _ in range(len(hidden_dims))]

    def _create_transformer_components(self, config, transformer_wte, blocks):
        """Create the transformer module dictionary."""
        ln_f = (
            nn.Identity()
        )  # RMSNorm() if config.use_rms_norm else LayerNorm(config.n_embd, bias=config.bias)

        components = {
            "wte": transformer_wte,
            "drop": nn.Dropout(config.dropout),
            "h": blocks,
            "ln_f": ln_f,
        }

        # Add positional embeddings if not using relative or rotary encoding
        if not getattr(config, "use_relative_position_encoding", False) and not getattr(
            config, "use_rotary_embeddings", False
        ):
            components["wpe"] = nn.Embedding(config.block_size, config.n_embd)

        return nn.ModuleDict(components)

    def create_kv_cache(self, batch_size, max_seq_len=None, device=None, dtype=None):
        """Factory helper to create a KV cache aligned with this model's architecture."""
        num_layers = len(self.transformer.h)
        num_kv_heads = getattr(self.config, "n_kv_head", self.config.n_head) or self.config.n_head
        head_dim = self.config.n_embd // self.config.n_head
        max_seq_len = max_seq_len or self.config.block_size
        device = device or next(self.parameters()).device
        dtype = dtype or next(self.parameters()).dtype
        return KVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def _initialize_rotary_cache(self, config, transformer_wte):
        """Precompute rotary embeddings once and share across all attention blocks.

        Be robust to different embedding implementations (e.g., `nn.Embedding`,
        `EmbeddingAdapter`, `LoRAEmbeddingAdapter`) where the token embedding
        may not expose a direct `.weight` attribute.
        """
        head_dim = config.n_embd // config.n_head
        cache_multiplier = max(1, getattr(config, "rotary_cache_multiplier", 2))
        seq_len = config.block_size * cache_multiplier

        # Infer device from provided embedding module or fall back to model params
        device = infer_device_from_embedding(transformer_wte, self)
        cos, sin = self._precompute_rotary_embeddings(seq_len, head_dim, device)
        self.register_buffer("rotary_cos", cos, persistent=False)
        self.register_buffer("rotary_sin", sin, persistent=False)
        self._rotary_cache_seq_len = seq_len
        self._rotary_head_dim = head_dim

    def _ensure_rotary_cache_capacity(self, required_len, device):
        """Grow or relocate the rotary cache if sequence/device requirements change."""
        if (
            hasattr(self, "rotary_cos")
            and required_len <= self._rotary_cache_seq_len
            and self.rotary_cos.device == device
        ):
            return

        new_len = (
            required_len
            if not hasattr(self, "_rotary_cache_seq_len")
            else max(required_len, self._rotary_cache_seq_len * 2)
        )
        cos, sin = self._precompute_rotary_embeddings(new_len, self._rotary_head_dim, device)
        self.rotary_cos = cos
        self.rotary_sin = sin
        self._rotary_cache_seq_len = new_len
        self._propagate_rotary_cache_to_blocks()

    def _propagate_rotary_cache_to_blocks(self):
        """Share rotary cache tensors with all transformer blocks."""
        if not hasattr(self, "rotary_cos"):
            return
        for block in self.transformer.h:
            if hasattr(block, "set_rotary_cache"):
                block.set_rotary_cache(self.rotary_cos, self.rotary_sin)

    def _get_rotary_slice(self, seq_len, offset, device):
        cos = self.rotary_cos[:, offset : offset + seq_len].to(device=device)
        sin = self.rotary_sin[:, offset : offset + seq_len].to(device=device)
        return cos, sin

    def _precompute_rotary_embeddings(self, seq_len, head_dim, device=None, base=10000):
        """Precompute rotary embeddings following the nanochat reference implementation."""
        if device is None:
            device = infer_device_from_embedding(self.transformer.wte, self)
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        # Only cast to bf16 on CUDA; MPS/CPU matmul may not accept bf16 accumulators
        # if (
        #     self.config.input_dtype == "bfloat16"
        #     and isinstance(device, torch.device)
        #     and device.type == "cuda"
        # ):
        cos, sin = cos.bfloat16(), sin.bfloat16()

        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        return cos, sin

    def _setup_event_embeddings(self, config):
        """Setup event type embeddings if enabled."""
        if self.use_event_names:
            self.transformer.update(
                {"ete": nn.Embedding(self.event_vocab_size, self.event_embedding_dim)}
            )

            if self.event_embedding_dim > 0:
                self.embedding_projection = nn.Linear(
                    config.n_embd + self.event_embedding_dim, config.n_embd
                )

    def _create_language_model_head(self, config, transformer_wte):
        """Create the language model head (lm_head)."""
        if config.use_embedding_adapter and config.weight_tying:
            self._create_prediction_head(config, transformer_wte)
        else:
            self._create_linear_head(config, transformer_wte)

    def _create_prediction_head(self, config, transformer_wte):
        """Create prediction head that uses embedding adapter."""
        if self.verbose:
            print("Using embedding adapter for lm_head")

        # import numpy as np
        # custom_embeddings = np.load(config.wte_from_numpy_path)
        # custom_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(custom_embeddings))
        # custom_embeddings = CustomEmbeddings(custom_embeddings, transformer_wte)

        self.lm_head = PredictionHead(
            embedding_adapter=transformer_wte,
            # custom_embeddings=custom_embeddings,
            chunk_size=config.logit_chunk_size,
            wte_from_numpy_path=config.wte_from_numpy_path,
            verbose=self.verbose,
            trainable_embeddings=config.embeddings_trainable,
            use_adapter_for_logits=config.use_embedding_adapter_for_logits,
            dtype=config.input_dtype
        )

    def _create_linear_head(self, config, transformer_wte):
        """Create standard linear projection head."""
        if self.verbose:
            print("Using standard linear projection for lm_head")

        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=getattr(config, "bias", False)
        )

        if config.weight_tying:
            if self.verbose:
                print("Weight tying enabled")
            self.lm_head.LLMC_SKIP_INIT = 1
            self.lm_head.weight = transformer_wte.weight

    def _initialize_weights(self, config):
        """Initialize model weights."""
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(DEFAULT_RANDOM_SEED)
        self.apply(self._init_weights)

        if getattr(config, "use_muparam_init", False):
            # Zero out LM head weights
            if hasattr(self, "lm_head") and isinstance(self.lm_head, nn.Linear):
                torch.nn.init.zeros_(self.lm_head.weight)

            # Zero out projection heads
            for block in self.transformer.h:
                if hasattr(block, "mlp") and hasattr(block.mlp, "c_proj"):
                    torch.nn.init.zeros_(block.mlp.c_proj.weight)
                if hasattr(block, "attn") and hasattr(block.attn, "c_proj"):
                    torch.nn.init.zeros_(block.attn.c_proj.weight)

            # Reinitialize rotary cache to default dtype/device
            if hasattr(self, "rotary_cos"):
                head_dim = self.config.n_embd // self.config.n_head
                cos, sin = self._precompute_rotary_embeddings(
                    self._rotary_cache_seq_len, head_dim, self.rotary_cos.device
                )
                self.rotary_cos = cos
                self.rotary_sin = sin

        # Cast embeddings to bf16 on CUDA for memory savings
        # device_for_embed = infer_device_from_embedding(self.transformer.wte, self)
        # if device_for_embed.type == "cuda" and self.config.input_dtype == "bfloat16":
        # if self.config.input_dtype == "bfloat16":
        #     self.transformer.wte = self.transformer.wte.to(dtype=torch.bfloat16)
        #     self.transformer.ete = self.transformer.ete.to(dtype=torch.bfloat16)

    def _setup_embeddings(self, config):
        """Load and configure embeddings."""
        if config.use_direct_embeddings:
            return

        wte_ref = self._get_embedding_reference(config)

        if config.wte_from_numpy_path:
            self._load_numpy_embeddings(config, wte_ref)

        if config.freeze_wte:
            self._freeze_embeddings(wte_ref)

    def _get_embedding_reference(self, config):
        """Get reference to the appropriate embedding layer."""
        return (
            self.transformer.wte if not config.use_embedding_adapter else self.transformer.wte.wte
        )

    def _load_numpy_embeddings(self, config, wte_ref):
        """Load embeddings from numpy file."""
        import numpy as np

        if self.verbose:
            print(f"Loading token embeddings from {config.wte_from_numpy_path}")

        wte_weights = np.load(config.wte_from_numpy_path)
        expected_shape = (config.vocab_size, wte_ref.weight.shape[1])

        if wte_weights.shape != expected_shape:
            warnings.warn(
                f"Embedding shape mismatch: expected {expected_shape}, got {wte_weights.shape}"
            )
            return

        wte_ref.weight.data.copy_(torch.from_numpy(wte_weights))

    def _freeze_embeddings(self, wte_ref):
        """Freeze token embeddings."""
        if self.verbose:
            print("Freezing token embeddings")
        wte_ref.weight.requires_grad = False

    def _print_model_summary(self, config):
        """Print detailed model summary and statistics."""
        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")
        self._print_basic_summary(config)
        self._print_layer_statistics(config)
        self._print_parameter_details()

    def _print_basic_summary(self, config):
        """Print basic model configuration summary."""
        print("\\n====== Model Summary ======")
        print(f"- Random seed: {DEFAULT_RANDOM_SEED}")
        print(f"- Architecture: GPT with {'MoE layers' if config.n_exp > 1 else 'standard layers'}")
        print(f"- Layers: {config.n_layer}")
        print(f"- Heads: {config.n_head}")
        print(f"- Embedding dim: {config.n_embd}")
        print(f"- Vocab size: {config.vocab_size}")
        print(f"- Block size: {config.block_size}")
        norm_str = (
            "RMSNorm (functional, no trainable gamma/beta)"
            if getattr(config, "use_rms_norm", False)
            else "LayerNorm (learnable gamma/beta)"
        )
        print(f"- Normalization (blocks): {norm_str}")
        # Attention always applies parameter-free RMSNorm to Q and K for stability
        print("- Attention QK RMSNorm: Enabled (parameter-free)")

        self._print_adapter_info(config)
        self._print_moe_info(config)
        self._print_embedding_info(config)

    def _print_adapter_info(self, config):
        """Print embedding adapter information."""
        print(f"- Embedding adapter: {'Yes' if config.use_embedding_adapter else 'No'}")

        if config.use_embedding_adapter and config.use_lora_adapter:
            print("  - Type: LoRA adapter")
            print(f"  - LoRA rank: {config.lora_rank}")
            print(f"  - LoRA alpha: {config.lora_alpha}")
            print(f"  - LoRA layers: {config.lora_num_layers}")

            if hasattr(self.transformer.wte, "get_num_lora_parameters"):
                lora_params = self.transformer.wte.get_num_lora_parameters()
                print(f"  - LoRA parameters: {lora_params:,}")
        elif config.use_embedding_adapter:
            print("  - Type: Dense MLP adapter")
            adapter_norm = (
                "RMSNorm (functional)"
                if getattr(config, "use_rms_norm", False)
                else "L2Norm (param-free)"
            )
            print(f"  - Adapter norm: {adapter_norm}")

    def _print_moe_info(self, config):
        """Print Mixture of Experts information."""
        if config.n_exp > 1:
            print(f"- Number of experts: {config.n_exp}")
            print(f"- Top-k routing: {config.top_k}")
            print(f"- MoE stride: {config.stride}")

    def _print_embedding_info(self, config):
        """Print embedding configuration information."""
        print(f"- Weight tying: {'Yes' if config.weight_tying else 'No'}")
        print(f"- Custom embeddings: {'Yes' if config.wte_from_numpy_path else 'No'}")

        if config.wte_from_numpy_path:
            print(f"  - Source: {config.wte_from_numpy_path}")

        print(f"- Frozen embeddings: {'Yes' if config.freeze_wte else 'No'}")

        smoothing_status = "(disabled)" if config.label_smoothing == 0.0 else "(enabled)"
        print(f"- Label smoothing: {config.label_smoothing:.3f} {smoothing_status}")

        temp_status = "(disabled)" if config.temperature_scaling == 1.0 else "(enabled)"
        print(f"- Temperature scaling: {config.temperature_scaling:.3f} {temp_status}")

    def _print_layer_statistics(self, config):
        """Print detailed layer statistics."""
        print("\\n====== Layer Statistics ======")

        param_counts = self._count_parameters_by_type()
        total_params = param_counts["total"]

        self._print_parameter_breakdown(param_counts, total_params, config)
        self._print_per_layer_stats(config)
        self._print_memory_usage()

    def _count_parameters_by_type(self):
        """Count parameters by type and return statistics."""
        counts = {
            "total": 0,
            "embedding": 0,
            "attention": 0,
            "mlp": 0,
            "layernorm": 0,
            "router": 0,
            "moe": 0,
            "lora": 0,
            "other": 0,
        }

        for name, param in self.named_parameters():
            if param.requires_grad:
                param_count = param.numel()
                counts["total"] += param_count

                if "wte" in name or "wpe" in name:
                    counts["embedding"] += param_count
                elif "lora_A" in name or "lora_B" in name:
                    counts["lora"] += param_count
                elif "attn" in name:
                    counts["attention"] += param_count
                elif "mlp" in name and "experts" not in name:
                    counts["mlp"] += param_count
                elif "ln" in name:
                    counts["layernorm"] += param_count
                elif "router" in name:
                    counts["router"] += param_count
                elif "experts" in name:
                    counts["moe"] += param_count
                else:
                    counts["other"] += param_count

        return counts

    def _print_parameter_breakdown(self, counts, total_params, config):
        """Print parameter count breakdown by type."""
        print(f"- Total trainable parameters: {total_params:,}")
        print(
            f"- Embeddings: {counts['embedding']:,} ({counts['embedding']/total_params*100:.2f}%)"
        )
        print(
            f"- Attention layers: {counts['attention']:,} ({counts['attention']/total_params*100:.2f}%)"
        )
        print(f"- Standard MLP layers: {counts['mlp']:,} ({counts['mlp']/total_params*100:.2f}%)")
        if getattr(config, "use_rms_norm", False):
            print("- Normalization params: 0 (RMSNorm is functional)")
        else:
            print(
                f"- LayerNorm layers: {counts['layernorm']:,} ({counts['layernorm']/total_params*100:.2f}%)"
            )

        if counts["lora"] > 0:
            print(f"- LoRA parameters: {counts['lora']:,} ({counts['lora']/total_params*100:.2f}%)")

        if counts["moe"] > 0:
            print(f"- MoE experts: {counts['moe']:,} ({counts['moe']/total_params*100:.2f}%)")
            print(
                f"- Router components: {counts['router']:,} ({counts['router']/total_params*100:.2f}%)"
            )

        if counts["other"] > 0:
            print(
                f"- Other parameters: {counts['other']:,} ({counts['other']/total_params*100:.2f}%)"
            )

    def _print_per_layer_stats(self, config):
        """Print parameter count for each layer."""
        print("\\n- Parameters per layer:")

        for i in range(config.n_layer):
            layer_name = f"h.{i}"
            layer_params = sum(
                p.numel()
                for name, p in self.named_parameters()
                if layer_name in name and p.requires_grad
            )
            is_moe = config.n_exp > 1 and (i + 1) % config.stride == 0
            moe_suffix = " (MoE)" if is_moe else ""
            print(f"  - Layer {i}: {layer_params:,} parameters{moe_suffix}")

    def _print_memory_usage(self):
        """Print estimated memory usage."""
        param_size_bytes = sum(
            p.nelement() * p.element_size() for p in self.parameters() if p.requires_grad
        )
        print(f"\\n- Approx. memory for parameters: {param_size_bytes / (1024**2):.2f} MB")

    def _print_parameter_details(self):
        """Print detailed parameter information table."""
        print("\\n====== Parameter Details ======")
        print(f"{'Parameter Name':<60} {'Shape':<20} {'Size':<12} {'Requires Grad':<15}")
        print("-" * 110)

        for name, param in sorted(self.named_parameters()):
            shape_str = str(list(param.shape))
            print(f"{name:<60} {shape_str:<20} {param.numel():<12,} {str(param.requires_grad):<15}")

        print("-" * 110)

    def _create_transformer_blocks(self, config, shared_relative_biases=None):
        """
        Create transformer blocks. Can be overridden by subclasses to use different layer types.

        Args:
            config: Model configuration
            shared_relative_biases: Optional shared bias terms for relative attention

        Returns:
            nn.ModuleList of transformer blocks
        """
        if config.n_exp == 1:
            # create normal transformer blocks
            if getattr(config, "use_relative_position_encoding", False):
                blocks = nn.ModuleList(
                    [
                        Block(
                            config,
                            layer_idx=i,
                            shared_relative_biases=shared_relative_biases,
                        )
                        for i in range(config.n_layer)
                    ]
                )
            else:
                blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        else:
            # create transformer blocks, placing an MoE block every <stride>
            # layers
            blocks = []
            for i in range(config.n_layer):
                use_moe = (i + 1) % config.stride == 0
                if getattr(config, "use_relative_position_encoding", False):
                    blocks.append(
                        Block(
                            config,
                            use_moe=use_moe,
                            layer_idx=i,
                            shared_relative_biases=shared_relative_biases,
                        )
                    )
                else:
                    blocks.append(Block(config, use_moe=use_moe))
            blocks = nn.ModuleList(blocks)

        return blocks

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.

        Only includes parameters that require gradients unless non_embedding=False.
        """
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding:
            # Only subtract position embeddings if they exist and require gradients
            # (wpe doesn't exist when using relative positional encoding)
            if hasattr(self.transformer, "wpe") and self.transformer.wpe.weight.requires_grad:
                n_params -= self.transformer.wpe.weight.numel()
        return n_params

    @torch.no_grad()
    def _init_weights(self, module):
        # optionally use switch transformer-style initialization
        # see page 10 for switch init explanation:
        # https://arxiv.org/abs/2101.03961
        if isinstance(module, nn.Linear):
            if self.config.use_switch_tfm_init:
                scale = self.config.switch_tfm_init_scale

                # linear layers have flipped dimensions in torch
                # size of weights is [out_dim, in_dim]
                w_fan_in = module.weight.shape[-1]
                w_std = (scale / w_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=w_std,
                    a=-2 * w_std,
                    b=2 * w_std,
                )
                if self.verbose:
                    print(
                        f"Initializing {module.__class__.__name__} with truncated normal distribution, shape: {module.weight.shape}"
                    )
            else:
                # std = (
                #     0.02
                #     if not hasattr(module, "LLMC_RESIDUAL_SCALE_FLAG")
                #     else 0.02 / math.sqrt(2 * self.config.n_layer)
                # )
                # std = 0.01
                # std = 0.02 / math.sqrt(2 * self.config.n_layer)
                if self.verbose:
                    print(
                        f"Initializing {module.__class__.__name__} with xavier_uniform distribution, shape: {module.weight.shape}"
                    )
                # perform standard (normal) initialization of weights
                # torch.nn.init.normal_(
                #     module.weight, mean=0.0, std=std, generator=self.init_rng
                # )
                torch.nn.init.xavier_uniform_(module.weight, gain=1.0)

            # always initialize bias to zero
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, MLPExperts):
            # we have to init expert weights manually because
            # nn.Parameter is not a type of module in torch
            if self.config.use_switch_tfm_init:
                scale = self.config.switch_tfm_init_scale

                c_fc_fan_in = module.c_fc.shape[-2]
                c_fc_std = (scale / c_fc_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.c_fc,
                    mean=0.0,
                    std=c_fc_std,
                    a=-2 * c_fc_std,
                    b=2 * c_fc_std,
                )

                c_proj_fan_in = module.c_proj.shape[-2]
                c_proj_std = (scale / c_proj_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.c_proj,
                    mean=0.0,
                    std=c_proj_std,
                    a=-2 * c_proj_std,
                    b=2 * c_proj_std,
                )
            else:
                # perform standard (normal) initialization of weights
                torch.nn.init.normal_(module.c_fc, mean=0.0, std=0.02, generator=self.init_rng)
                torch.nn.init.normal_(module.c_proj, mean=0.0, std=0.02, generator=self.init_rng)

            # bias is always initialized to zero
            if module.fc_bias is not None:
                torch.nn.init.zeros_(module.fc_bias)
                torch.nn.init.zeros_(module.proj_bias)

        elif isinstance(module, nn.Embedding):
            # just use standard initialization scheme for embedding always
            if module.weight.requires_grad:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

        elif isinstance(module, EmbeddingAdapter):
            # Special initialization for embedding adapter layers
            if self.verbose:
                print("Initializing EmbeddingAdapter module")
            # The adapter layers will be initialized by their respective Linear
            # modules

            # If we have a residual projection, initialize with small values
            if hasattr(module, "residual_proj") and module.residual_proj is not None:
                if self.verbose:
                    print("Initializing residual projection with small values")
                # Initialize the residual projection with small values
                torch.nn.init.normal_(
                    module.residual_proj.weight,
                    mean=0.0,
                    std=0.02 / math.sqrt(module.n_embd),
                    generator=self.init_rng,
                )

        elif isinstance(module, LoRAEmbeddingAdapter):
            # Special initialization for LoRA embedding adapter
            if self.verbose:
                print("Initializing LoRAEmbeddingAdapter module")
            # LoRA layers will be initialized by their own
            # reset_lora_parameters method

        elif isinstance(module, LoRALayer):
            # LoRA layers have their own initialization in __init__
            if self.verbose:
                print(
                    f"LoRA layer initialized: rank={module.rank}, in_features={module.in_features}, out_features={module.out_features}"
                )
            # The LoRA parameters are already initialized in __init__ with
            # reset_lora_parameters()

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         # apply special scaled init to the residual projections, per GPT-2 paper
    #         std = (
    #             0.02
    #             if not hasattr(module, "LLMC_RESIDUAL_SCALE_FLAG")
    #             else 0.02 / math.sqrt(2 * self.config.n_layer)
    #         )
    #         # we want to skip initializing lm_head, which shares parameters with wte
    #         # and wte was already initialized down below during the Embedding init
    #         if not hasattr(module, "LLMC_SKIP_INIT"):
    #             torch.nn.init.normal_(
    #                 module.weight, mean=0.0, std=std, generator=self.init_rng
    #             )
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         torch.nn.init.normal_(
    #             module.weight, mean=0.0, std=0.02, generator=self.init_rng
    #         )

    def forward(
        self,
        idx,
        targets=None,
        return_logits=True,
        event_idx=None,
        segment_ids=None,
        kv_cache=None,
    ):
        """
        Forward pass through the GPT model.

        Args:
            idx (torch.Tensor): Input token indices of shape (batch_size, sequence_length)
            targets (torch.Tensor, optional): Target token indices for loss computation
            return_logits (bool): Whether to return logits or just compute loss
            event_idx (torch.Tensor, optional): Event indices for event-aware modeling
            segment_ids (torch.Tensor, optional): Segment IDs for multi-segment sequences
            kv_cache (KVCache, optional): Cache for streaming inference

        Returns:
            torch.Tensor or tuple: If targets provided, returns loss and optionally logits.
                                 Otherwise returns logits only.
        """
        device = idx.device
        b, t = self._validate_input(idx)

        # Get token embeddings with optional event embeddings
        tok_emb = self._get_token_embeddings(idx, event_idx)
        # if self.config.debug:
        # print('tok_emb dtype', tok_emb.dtype)

        # Validate sequence length
        self._validate_sequence_length(t)

        if getattr(self.config, "use_rotary_embeddings", False):
            self._ensure_rotary_cache_capacity(t, device)

        rotary_offset = (
            kv_cache.get_pos()
            if kv_cache is not None and getattr(self.config, "use_rotary_embeddings", False)
            else 0
        )
        if getattr(self.config, "use_rotary_embeddings", False):
            self._ensure_rotary_cache_capacity(rotary_offset + t, device)

        # Add positional information and apply dropout
        x = self._add_positional_embeddings(tok_emb, t, device)

        # Forward through transformer layers
        x = self._forward_through_blocks(
            x,
            segment_ids,
            kv_cache=kv_cache,
            rotary_offset=rotary_offset,
        )

        # Compute logits and loss
        logits, loss = self._compute_output(x, targets)

        # Apply return_logits flag
        if not return_logits:
            logits = None

        return logits, loss

    def _validate_input(self, idx):
        """Validate input and return batch size and sequence length."""
        if hasattr(self.config, "use_direct_embeddings") and self.config.use_direct_embeddings:
            if idx.dim() == 3:
                # Direct embeddings passed
                b, t, _ = idx.size()
            else:
                raise ValueError(
                    "Model configured for direct embeddings but received token IDs. "
                    "When use_direct_embeddings=True, the input should be embeddings."
                )
        else:
            # Normal mode: lookup token embeddings from token IDs
            b, t = idx.size()
        return b, t

    def _get_token_embeddings(self, idx, event_idx):
        """Get token embeddings, optionally combining with event embeddings."""
        tok_emb = self.transformer.wte(idx)

        if self.use_event_names and event_idx is not None:
            tok_emb = self._add_event_embeddings(tok_emb, event_idx)

        return tok_emb

    def _add_event_embeddings(self, tok_emb, event_idx):
        """Add event embeddings to token embeddings."""
        event_emb = self.transformer.ete(event_idx)
        combined_emb = torch.cat([tok_emb, event_emb], dim=-1)
        return self.embedding_projection(combined_emb)

    def _validate_sequence_length(self, t):
        """Ensure sequence length is within block size."""
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

    def _add_positional_embeddings(self, tok_emb, t, device):
        """Add positional embeddings based on encoding type."""
        if getattr(self.config, "use_relative_position_encoding", False) or getattr(
            self.config, "use_rotary_embeddings", False
        ):
            # For relative or rotary encoding, no absolute positional embeddings are added here
            x = tok_emb
        else:
            # Add absolute positional embeddings for standard GPT
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            x = tok_emb + pos_emb

        x = self._apply_embedding_norm_if_needed(x)
        return self.transformer.drop(x)

    def _apply_embedding_norm_if_needed(self, embeddings):
        """Optionally apply RMS norm to embeddings before transformer blocks."""
        if self.embedding_rms_norm is not None:
            return self.embedding_rms_norm(embeddings)
        return embeddings

    def _forward_through_blocks(self, x, segment_ids, kv_cache=None, rotary_offset=0):
        """Forward through transformer blocks."""
        use_relative_encoding = getattr(self.config, "use_relative_position_encoding", False)
        use_rotary = getattr(self.config, "use_rotary_embeddings", False)
        rotary_embeddings = None
        if use_rotary:
            rotary_embeddings = self._get_rotary_slice(x.size(1), rotary_offset, x.device)

        for block in self.transformer.h:
            if use_relative_encoding:
                x = block(
                    x,
                    segment_ids=segment_ids,
                    kv_cache=kv_cache,
                )
            else:
                x = block(
                    x,
                    kv_cache=kv_cache,
                    rotary_embeddings=rotary_embeddings,
                )

        return self.transformer.ln_f(x)

    def _compute_output(self, x, targets):
        """Compute logits and loss based on whether targets are provided."""
        if targets is not None:
            return self._compute_training_output(x, targets)
        else:
            return self._compute_inference_output(x)

    def _compute_training_output(self, x, targets):
        """Compute logits and loss for training mode."""
        logits = self.lm_head(x)
        if getattr(self.config, "use_logit_softcap", False):
            softcap = getattr(self.config, "logit_softcap_value", 15.0) or 15.0
            logits = softcap * torch.tanh(logits / softcap)

        if targets.dim() == 1:
            # Only predicting last token
            logits = logits[:, -1, :]
            loss = label_smoothing_cross_entropy(
                logits,
                targets,
                self.config.label_smoothing,
                self.config.temperature_scaling,
            )
        else:
            # Predict all next tokens
            loss = label_smoothing_cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                self.config.label_smoothing,
                self.config.temperature_scaling,
            )

        # Check for NaN loss
        if torch.isnan(loss).any():
            print("WARNING: NaN detected in loss")

        # Add auxiliary losses for MoE models
        loss = self._add_auxiliary_losses(loss)

        return logits, loss

    def _compute_inference_output(self, x):
        """Compute logits for inference mode (only last position)."""
        last_hidden = x[:, [-1], :]  # Preserve time dimension
        logits = self.lm_head(last_hidden, training=False)
        if getattr(self.config, "use_logit_softcap", False):
            softcap = getattr(self.config, "logit_softcap_value", 15.0) or 15.0
            logits = softcap * torch.tanh(logits / softcap)
        return logits, None

    def _add_auxiliary_losses(self, loss):
        """Add auxiliary losses for MoE models."""
        if self.config.n_exp > 1:
            if self.config.use_aux_loss:
                loss += self.config.aux_loss_weight * MANAGER.aggregate_aux_loss()
                MANAGER.reset_aux_loss()

            if self.config.use_router_z_loss:
                loss += self.config.router_z_loss_weight * MANAGER.aggregate_router_z_loss()
                MANAGER.reset_router_z_loss()

        return loss

    @classmethod
    def to_inference(cls, model, ptdtype):
        """
        Converts a GPT or Mamba model to inference mode for more efficient generation.

        In inference mode:
        1. Only the last position is used for prediction
        2. The lm_head is replaced with a more efficient implementation

        Args:
            model: The GPT or Mamba model to convert (may be wrapped/optimized)

        Returns:
            Modified model optimized for inference (preserves original model type)
        """

        def unwrap_model(wrapped_model):
            """
            Recursively unwrap model from optimization/compilation wrappers.

            Handles common PyTorch wrappers like:
            - OptimizedModule (torch.compile)
            - DistributedDataParallel
            - DataParallel
            - ScriptModule
            - Custom wrappers with 'module' or '_orig_mod' attributes
            """
            current = wrapped_model

            # Keep unwrapping until we find the actual model
            max_unwrap_depth = 10  # Prevent infinite loops
            depth = 0

            while depth < max_unwrap_depth:
                # Check for common unwrapping patterns
                if hasattr(current, "_orig_mod"):
                    # OptimizedModule and torch.compile wrappers
                    current = current._orig_mod
                elif hasattr(current, "module"):
                    # DDP, DataParallel, and other common wrappers
                    current = current.module
                elif hasattr(current, "_modules") and len(current._modules) == 1:
                    # Some wrappers store the model as the only module
                    module_name = next(iter(current._modules.keys()))
                    potential_model = current._modules[module_name]
                    if hasattr(potential_model, "_create_transformer_blocks"):
                        current = potential_model
                    else:
                        break
                else:
                    # No more unwrapping possible
                    break

                depth += 1

                # Check if we've found a valid model
                if hasattr(current, "_create_transformer_blocks"):
                    break

            return current

        class LastTokenLMHead(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, training=True):
                # Only use the last position
                return x[:, -1, :]

        # Handle the torch Generator issue safely
        # Create a state dict without the unpicklable generator
        state_dict = model.state_dict()

        # Unwrap the model to get to the actual GPT/Mamba instance
        raw_model = unwrap_model(model)

        # Get config from the unwrapped model
        if hasattr(raw_model, "config"):
            config = raw_model.config
        else:
            raise ValueError(
                f"Could not find config in model. Unwrapped model type: {type(raw_model).__name__}. "
                "Make sure the model is a GPT or Mamba instance."
            )

        # Determine the actual model type based on layer structure, not class
        # inheritance
        def determine_model_type(model, state_dict):
            """
            Determine if this is actually a GPT or Mamba model by checking:
            1. The actual layer types in the model
            2. The parameter names in the state_dict
            """
            # First, check the state_dict for characteristic parameters
            if any("mamba." in key for key in state_dict.keys()) or any(
                ".mamba" in key for key in state_dict.keys()
            ):
                # Has Mamba-specific parameters
                return Mamba

            elif any("attn.c_attn" in key for key in state_dict.keys()) or any(
                "attn.c_q" in key for key in state_dict.keys()
            ):
                # Has GPT-specific attention parameters
                return GPT

                # If state_dict check is inconclusive, check actual layer types
            if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                if len(model.transformer.h) > 0:
                    first_layer = model.transformer.h[0]

                    if hasattr(first_layer, "mamba"):
                        return Mamba
                    elif hasattr(first_layer, "attn"):
                        return GPT

            # Fallback to class type, but prefer checking inheritance properly
            if isinstance(model, Mamba):
                return Mamba

            elif isinstance(model, GPT):
                return GPT

            else:
                return type(model)

        model_type = determine_model_type(raw_model, state_dict)

        # Validate that it's a supported model type
        if not hasattr(model_type, "_create_transformer_blocks"):
            raise ValueError(
                f"Unsupported model type: {model_type.__name__}. "
                f"Expected GPT or Mamba model with _create_transformer_blocks method. "
                f"Original wrapped model type: {type(model).__name__}. "
                f"If this is a wrapped model, the unwrapping may have failed."
            )

        # Create a new model instance with the same type and config
        try:
            model_copy = model_type(config, inference=True)
        except (TypeError, ValueError) as e:
            raise RuntimeError(
                f"Failed to create new {model_type.__name__} instance for inference: {e}"
            )

        # Load the state dict into the new model
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        try:
            model_copy.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load state_dict into {model_type.__name__} model. "
                f"This usually means the model architecture has changed. "
                f"Original error: {e}"
            )

        # Replace with last-token-only version
        model_copy.lm_head = LastTokenLMHead()

        # Set model to eval mode
        model_copy.eval()

        print(
            f"Model converted to inference mode. Using only last token for prediction. "
            f"Original type: {model_type.__name__} (was wrapped as {type(model).__name__})"
        )

        if ptdtype == torch.bfloat16:
            print('convert to bfloat16 for to_inference')
            model_copy.transformer.wte = model_copy.transformer.wte.to(torch.float32)

        return model_copy

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        event_idx=None,
        segment_ids=None,
    ):
        """
        Generate text token by token.

        Args:
            idx: Starting token sequence
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature for generation (different from config.temperature_scaling)
            top_k: Number of top tokens to sample from
            event_idx: Optional event type indices

        Returns:
            Generated token sequence

        Note:
            The 'temperature' parameter here is for sampling during generation and is applied
            after the model forward pass. This is different from 'config.temperature_scaling'
            which is applied during training loss calculation to calibrate the model's confidence.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Truncate sequence if needed
            idx_cond = (
                idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            )

            # Also truncate event_idx if provided
            event_idx_cond = None
            if self.use_event_names and event_idx is not None:
                event_idx_cond = (
                    event_idx
                    if event_idx.size(1) <= self.config.block_size
                    else event_idx[:, -self.config.block_size :]
                )

            # Also truncate segment_ids if provided
            segment_ids_cond = None
            if (
                getattr(self.config, "use_relative_position_encoding", False)
                and segment_ids is not None
            ):
                segment_ids_cond = (
                    segment_ids
                    if segment_ids.size(1) <= self.config.block_size
                    else segment_ids[:, -self.config.block_size :]
                )

            # Get logits
            logits = self(idx_cond, event_idx=event_idx_cond, segment_ids=segment_ids_cond)
            if isinstance(logits, tuple):
                logits = logits[0]

            # Take only the last token's logits
            logits = logits[:, -1, :] / temperature

            # Apply top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

            # If we have event indices, we need to append the corresponding event type
            # This depends on your application logic - here we just repeat the
            # last event
            if self.use_event_names and event_idx is not None:
                event_idx_next = event_idx[:, -1:].clone()
                event_idx = torch.cat((event_idx, event_idx_next), dim=1)

        return idx


@dataclass
class ModelArgs:
    """
    Configuration arguments for Mamba model parameters.

    Defines the model architecture parameters including dimensions,
    state size, expansion factor, and convolution settings used
    by the Mamba state space model components.
    """

    d_model: int
    # n_layer: int
    # vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

        # if self.vocab_size % self.pad_vocab_size_multiple != 0:
        #     self.vocab_size += (
        #         self.pad_vocab_size_multiple
        #         - self.vocab_size % self.pad_vocab_size_multiple
        #     )


@dataclass
class MambaConfig(GPTConfig):
    """Configuration class for Mamba model parameters, inheriting from GPTConfig"""

    # Mamba-specific parameters
    d_state: int = 16  # State dimension for Mamba SSM
    d_conv: int = 4  # Convolution kernel size for Mamba
    expand: int = 2  # Expansion factor for Mamba inner dimension

    def pretty_print(self):
        # Get parent class pretty_print output
        parent_dict = super().pretty_print()
        # Add Mamba-specific parameters
        parent_dict.update(
            {
                "d_state": self.d_state,
                "d_conv": self.d_conv,
                "expand": self.expand,
            }
        )
        return parent_dict


class MambaLayer(nn.Module):
    """
    Mamba layer with feed-forward network and residual connections.
    Adapted from the original Mamba4Rec implementation for the current project structure.
    """

    def __init__(self, config, layer_idx=0, inference=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.inference = inference

        # Import Mamba here to handle potential import errors gracefully

        if self.inference:
            print("Using MambaBlockMinimal")
            self.mamba_block = MambaBlockMinimal(
                ModelArgsMinimal(
                    d_model=config.n_embd,
                    d_state=config.d_state,
                    d_conv=config.d_conv,
                    expand=config.expand,
                )
            )
        else:
            try:
                from mamba_ssm import Mamba
            except ImportError:
                raise ImportError(
                    "mamba_ssm is required for Mamba models. "
                    "Please install it with: pip install mamba_ssm"
                )
            print("Using Mamba from mamba_ssm")
            self.mamba_block = Mamba(
                d_model=config.n_embd,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
            )

        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = LayerNorm(config.n_embd, bias=config.bias)
        self.ffn = MambaFeedForward(config)

    def forward(self, x):
        """
        Forward pass through the Mamba layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: Output tensor after Mamba block processing and normalization
        """
        # Apply Mamba block
        hidden_states = self.mamba_block(x)

        # Apply dropout and layer norm with residual connection
        if self.config.n_layer == 1:
            # Single layer without residual connection
            hidden_states = self.layer_norm(self.dropout(hidden_states))
        else:
            # Multiple layers with residual connections
            hidden_states = self.layer_norm(self.dropout(hidden_states) + x)

        # Apply feed-forward network
        hidden_states = self.ffn(hidden_states)

        return hidden_states


class MambaFeedForward(nn.Module):
    """
    Feed-forward network for Mamba layers with residual connections.
    """

    def __init__(self, config):
        super().__init__()
        inner_size = config.n_embd * 4  # Standard 4x expansion

        self.w_1 = nn.Linear(config.n_embd, inner_size)
        self.w_2 = nn.Linear(inner_size, config.n_embd)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x):
        """
        Forward pass through the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: Output tensor with residual connection applied
        """
        # Feed-forward transformation
        hidden_states = self.w_1(x)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Layer norm with residual connection
        hidden_states = self.layer_norm(hidden_states + x)

        return hidden_states


class Mamba(GPT):
    """
    Mamba model with the same interface as GPT for seamless integration.
    Uses state space models instead of transformer attention.
    Inherits from GPT to reuse all initialization logic and features.
    """

    def __init__(self, config, verbose=False, inference=False):
        # Ensure we have MambaConfig or convert GPTConfig to MambaConfig
        if not isinstance(config, MambaConfig):
            # Convert GPTConfig to MambaConfig to ensure Mamba-specific
            # parameters
            mamba_config = MambaConfig()
            # Copy all GPTConfig attributes
            for key, value in config.__dict__.items():
                if hasattr(mamba_config, key):
                    setattr(mamba_config, key, value)
            config = mamba_config

        # Call parent constructor but override the block creation
        self._creating_mamba_layers = True
        super().__init__(config, verbose, inference)
        self._creating_mamba_layers = False

    def _create_transformer_blocks(self, config, shared_relative_biases=None):
        """
        Override method to create MambaLayer instead of Block.
        This method is called during GPT.__init__ to create the transformer layers.
        """
        if getattr(self, "_creating_mamba_layers", False):
            # Create MambaLayer instances instead of Block instances
            blocks = nn.ModuleList(
                [MambaLayer(config, i, self.inference) for i in range(config.n_layer)]
            )
            return blocks
        else:
            # Fallback to parent method for regular GPT
            return super()._create_transformer_blocks(config, shared_relative_biases)


class MambaBlockOptimized(nn.Module):
    """
    Optimized Mamba block implementation.

    A single Mamba block implementing the selective state space model
    as described in Figure 3 in Section 3.4 of the Mamba paper.
    This implementation provides efficient selective scan operations
    for sequence modeling tasks.
    """

    def __init__(self, args: ModelArgs):
        """
        Initialize a single Mamba block.

        Args:
            args (ModelArgs): Configuration arguments containing model dimensions
                and architectural parameters
        """
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), "n -> d n", d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (b, l, d) = x.shape

        # Ensure input tensor properties are consistent
        device = x.device
        dtype = x.dtype

        # Preemptive memory cleanup to prevent allocator corruption
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Check for NaN in input to prevent propagation
        if torch.isnan(x).any():
            print("WARNING: NaN detected in MambaBlock input")
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        # Clear intermediate tensor to save memory
        del x_and_res

        x = rearrange(x, "b l d_in -> b d_in l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d_in l -> b l d_in")

        x = F.silu(x)

        # Ensure tensor consistency before SSM
        x = x.to(device=device, dtype=dtype)

        y = self.ssm(x)

        # Clear x after SSM to save memory
        del x

        y = y * F.silu(res)

        # Clear res after use
        del res

        output = self.out_proj(y)

        return output

    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape

        # Ensure device/dtype consistency
        device = x.device
        dtype = x.dtype

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        # and is why Mamba is called **selective** state spaces)

        A = -torch.exp(self.A_log.float()).to(device=device)  # shape (d_in, n)
        D = self.D.float().to(device=device)

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(
            split_size=[self.args.dt_rank, n, n], dim=-1
        )  # delta: (b, l, dt_rank). B, C: (b, l, n)

        # Clear intermediate tensor
        del x_dbl

        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        # Ensure all tensors are on correct device and dtype
        delta = delta.to(device=device, dtype=dtype)
        B = B.to(device=device, dtype=dtype)
        C = C.to(device=device, dtype=dtype)

        y = self.selective_scan(
            x, delta, A, B, C, D
        )  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

        """
        b, l, d_in = u.shape
        n = A.shape[1]
        device, dtype = u.device, u.dtype

        # Discretize continuous parameters with memory optimization
        deltaA, deltaB_u = self._discretize_parameters(u, delta, A, B, l)

        # Perform selective scan
        y = self._perform_scan(deltaA, deltaB_u, C, b, l, d_in, n, device, dtype)

        # Add skip connection
        y = self._add_skip_connection(y, u, D, device, dtype)

        return y

    def _discretize_parameters(self, u, delta, A, B, l):
        """Discretize continuous parameters A and B with chunked processing."""
        chunk_size = min(32, l)
        deltaA_chunks = []
        deltaB_u_chunks = []

        for i in range(0, l, chunk_size):
            end_i = min(i + chunk_size, l)
            delta_chunk = delta[:, i:end_i]
            B_chunk = B[:, i:end_i]
            u_chunk = u[:, i:end_i]

            # Discretize A using zero-order hold (ZOH)
            deltaA_chunk = torch.exp(einsum(delta_chunk, A, "b l d_in, d_in n -> b l d_in n"))
            # Discretize B using simplified Euler
            deltaB_u_chunk = einsum(
                delta_chunk, B_chunk, u_chunk, "b l d_in, b l n, b l d_in -> b l d_in n"
            )

            deltaA_chunks.append(deltaA_chunk)
            deltaB_u_chunks.append(deltaB_u_chunk)

            # Clean up intermediate tensors
            del delta_chunk, B_chunk, u_chunk, deltaA_chunk, deltaB_u_chunk

        # Concatenate chunks
        deltaA = torch.cat(deltaA_chunks, dim=1)
        deltaB_u = torch.cat(deltaB_u_chunks, dim=1)

        del deltaA_chunks, deltaB_u_chunks
        return deltaA, deltaB_u

    def _perform_scan(self, deltaA, deltaB_u, C, b, l, d_in, n, device, dtype):
        """Perform the selective scan computation."""
        # Pre-allocate output tensors
        y = torch.zeros((b, l, d_in), device=device, dtype=dtype)
        x = torch.zeros((b, d_in, n), device=device, dtype=dtype)

        scan_chunk_size = min(16, l)

        for chunk_start in range(0, l, scan_chunk_size):
            chunk_end = min(chunk_start + scan_chunk_size, l)

            try:
                x, y = self._process_scan_chunk(x, y, deltaA, deltaB_u, C, chunk_start, chunk_end)
                self._manage_memory(chunk_start, scan_chunk_size)
            except RuntimeError as e:
                if "CUDA" in str(e) or "INTERNAL ASSERT FAILED" in str(e):
                    print(
                        f"CUDA error in selective_scan chunk {chunk_start}-{chunk_end}, processing on CPU"
                    )
                    x, y = self._process_chunk_on_cpu(
                        x, y, deltaA, deltaB_u, C, chunk_start, chunk_end, device, dtype
                    )
                else:
                    raise

        return y

    def _process_scan_chunk(self, x, y, deltaA, deltaB_u, C, chunk_start, chunk_end):
        """Process a single chunk of the scan operation."""
        for i in range(chunk_start, chunk_end):
            # State update: x(t+1) = deltaA * x(t) + deltaB_u
            x = deltaA[:, i] * x + deltaB_u[:, i]

            # Output computation: y(t) = C * x(t)
            y[:, i, :] = einsum(x, C[:, i, :], "b d_in n, b n -> b d_in")

        return x, y

    def _manage_memory(self, chunk_start, scan_chunk_size):
        """Manage GPU memory during scan processing."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if chunk_start % (scan_chunk_size * 8) == 0:  # Sync every 8 chunks
                torch.cuda.synchronize()

    def _process_chunk_on_cpu(
        self, x, y, deltaA, deltaB_u, C, chunk_start, chunk_end, device, dtype
    ):
        """Process a chunk entirely on CPU as fallback."""
        x_cpu = x.detach().cpu() if x.device.type == "cuda" else x.clone()

        try:
            for i in range(chunk_start, chunk_end):
                # Move tensors to CPU
                deltaA_cpu = deltaA[:, i].cpu() if deltaA.device.type == "cuda" else deltaA[:, i]
                deltaB_u_cpu = (
                    deltaB_u[:, i].cpu() if deltaB_u.device.type == "cuda" else deltaB_u[:, i]
                )
                C_cpu = C[:, i, :].cpu() if C.device.type == "cuda" else C[:, i, :]

                # Compute on CPU
                x_cpu = deltaA_cpu * x_cpu + deltaB_u_cpu
                y_step_cpu = einsum(x_cpu, C_cpu, "b d_in n, b n -> b d_in")

                # Store result
                if device.type == "cuda":
                    y[:, i, :] = y_step_cpu.to(device=device, dtype=dtype)
                else:
                    y[:, i, :] = y_step_cpu
        except RuntimeError as e:
            print(f"CPU fallback failed for chunk {chunk_start}-{chunk_end}: {e}")
            # Fill with fallback values
            for i in range(chunk_start, chunk_end):
                if i > 0:
                    y[:, i, :] = y[:, i - 1, :]
                else:
                    y[:, i, :] = torch.zeros_like(y[:, i, :])

        # Transfer x back to original device if needed
        if device.type == "cuda" and x.device.type == "cuda":
            return x_cpu.to(device=device, dtype=dtype)
        return x_cpu

    def _add_skip_connection(self, y, u, D, device, dtype):
        """Add skip connection with device safety."""
        try:
            D_safe = D.to(device=y.device, dtype=y.dtype)
            u_safe = u.to(device=y.device, dtype=y.dtype) if u.device != y.device else u
            return y + u_safe * D_safe
        except RuntimeError as e:
            if "CUDA" in str(e) or "INTERNAL ASSERT FAILED" in str(e):
                print(f"CUDA error in skip connection, keeping y as-is: {e}")
                return y  # Skip the skip connection to avoid further corruption
            else:
                raise
