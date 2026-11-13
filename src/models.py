"""Multimodal Fusion Transformer architecture combining FT-Transformer and cross-attention."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """FT-Transformer style tokenizer mapping continuous features to learned token embeddings.

    Transforms an input tensor of shape ``(B, num_features)`` into ``(B, 1 + num_features, d_model)``
    by applying per-feature affine projections and prepending a learnable ``[CLS]`` token.
    """

    def __init__(self, num_features: int, d_model: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(num_features, d_model))
        self.bias = nn.Parameter(torch.zeros(num_features, d_model))
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.05)

    def forward(self, x_tabular: torch.Tensor) -> torch.Tensor:
        if x_tabular.ndim != 2 or x_tabular.shape[1] != self.num_features:
            raise ValueError(
                f"Expected tabular shape (B, {self.num_features}), got {tuple(x_tabular.shape)}"
            )
        batch_size = x_tabular.shape[0]
        # (B, F, 1) * (1, F, D) + (1, F, D) -> (B, F, D)
        feature_tokens = x_tabular.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        return torch.cat([cls_tokens, feature_tokens], dim=1)


class TabularTransformerEncoder(nn.Module):
    """FT-Transformer encoder over tokenized tabular features and a prepended [CLS] token."""

    def __init__(
        self,
        num_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_tokenizer = FeatureTokenizer(num_features=num_features, d_model=d_model)
        self.norm_in = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

    def forward(self, x_tabular: torch.Tensor) -> torch.Tensor:
        tokens = self.feature_tokenizer(x_tabular)
        tokens = self.dropout(self.norm_in(tokens))
        return self.encoder(tokens)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding registered as a non-trainable buffer."""

    def __init__(self, d_model: int, max_len: int = 128) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]


class TextTransformerEncoder(nn.Module):
    """Self-contained transformer text encoder operating on deterministic token IDs."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        max_seq_len: int = 32,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_seq_len)
        self.norm_in = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids shape (B, L), got {tuple(input_ids.shape)}")
        pad_mask = input_ids == self.pad_token_id
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(self.norm_in(x))
        encoded = self.encoder(x, src_key_padding_mask=pad_mask)
        return encoded, pad_mask


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion with residual LayerNorm, gated projection, and classifier head."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.reverse_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_tab = nn.LayerNorm(d_model)
        self.norm_txt = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.tab_proj = nn.Linear(d_model, d_model)
        self.txt_proj = nn.Linear(d_model, d_model)
        self.interaction_proj = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        tab_tokens: torch.Tensor,
        txt_tokens: torch.Tensor,
        txt_pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Tabular tokens attend to non-padded text tokens.
        tab_attended, tab_to_txt_weights = self.cross_attention(
            query=tab_tokens,
            key=txt_tokens,
            value=txt_tokens,
            key_padding_mask=txt_pad_mask,
            need_weights=True,
        )
        tab_fused = self.norm_tab(tab_tokens + self.dropout(tab_attended))

        # Text tokens attend to tabular feature tokens.
        txt_attended, txt_to_tab_weights = self.reverse_attention(
            query=txt_tokens,
            key=tab_tokens,
            value=tab_tokens,
            need_weights=True,
        )
        txt_fused = self.norm_txt(txt_tokens + self.dropout(txt_attended))

        # Pool contextualized [CLS] summaries and apply gated + bilinear interaction fusion.
        z_tab = tab_fused[:, 0, :]
        z_txt = txt_fused[:, 0, :]
        gate = torch.sigmoid(self.gate_proj(torch.cat([z_tab, z_txt], dim=-1)))
        gated = gate * self.tab_proj(z_tab) + (1.0 - gate) * self.txt_proj(z_txt)
        interaction = self.interaction_proj(torch.cat([z_tab, z_txt, z_tab * z_txt], dim=-1))
        fused = self.fusion_norm(gated + interaction)

        logits = self.classifier(fused)
        attn_maps = {
            "tab_to_txt": tab_to_txt_weights,
            "txt_to_tab": txt_to_tab_weights,
        }
        return logits, attn_maps


class FusionTransformer(nn.Module):
    """End-to-end Multimodal Tabular + Text Fusion Transformer."""

    def __init__(
        self,
        num_features: int,
        vocab_size: int,
        num_classes: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        n_tabular_layers: int = 2,
        n_text_layers: int = 2,
        dim_feedforward: int = 128,
        max_seq_len: int = 32,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.tabular_encoder = TabularTransformerEncoder(
            num_features=num_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_tabular_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.text_encoder = TextTransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_text_layers,
            dim_feedforward=dim_feedforward,
            max_seq_len=max_seq_len,
            dropout=dropout,
            pad_token_id=pad_token_id,
        )
        self.fusion = CrossAttentionFusion(
            d_model=d_model,
            n_heads=n_heads,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        x_tabular: torch.Tensor,
        input_ids: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        tab_tokens = self.tabular_encoder(x_tabular)
        txt_tokens, txt_pad_mask = self.text_encoder(input_ids)
        logits, attn_maps = self.fusion(tab_tokens, txt_tokens, txt_pad_mask=txt_pad_mask)
        if return_attention:
            return logits, attn_maps
        return logits
