"""Configuration dataclass for the multimodal Fusion Transformer."""

from dataclasses import dataclass


@dataclass
class Config:
    """Hyperparameters, architecture dimensions, and training settings."""

    num_features: int = 10
    num_classes: int = 2
    d_model: int = 64
    n_heads: int = 4
    n_tabular_layers: int = 2
    n_text_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    max_seq_len: int = 32
    batch_size: int = 32
    num_epochs: int = 15
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    num_samples: int = 1200
    val_size: float = 0.15
    test_size: float = 0.15
    checkpoint_path: str = "outputs/fusion_transformer.pt"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.num_features < 1:
            raise ValueError(f"num_features must be >= 1, got {self.num_features}")
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {self.num_epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.val_size <= 0 or self.test_size <= 0 or (self.val_size + self.test_size) >= 1.0:
            raise ValueError("val_size and test_size must be positive and sum to < 1.0")


FusionConfig = Config
