"""Deterministic tokenization, multimodal dataset synthesis, and evaluation metrics."""

from dataclasses import dataclass
import logging
import random
import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
from torch.utils.data import Dataset

_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")

_ADVERSE_CLAUSES: List[Tuple[str, float]] = [
    ("repeated late notices reported", 1.35),
    ("escalated compliance flag active", 1.55),
    ("unexpected cashflow shortfall observed", 1.45),
    ("elevated chargeback volume detected", 1.30),
    ("pending audit warning issued", 1.50),
    ("missed covenant milestone recorded", 1.40),
]

_MODERATE_CLAUSES: List[Tuple[str, float]] = [
    ("minor billing variance noted", 0.55),
    ("delayed invoice settlement pending", 0.65),
    ("tight operating margin observed", 0.50),
    ("temporary liquidity dip flagged", 0.60),
]

_MITIGATING_CLAUSES: List[Tuple[str, float]] = [
    ("verified collateral coverage confirmed", -1.30),
    ("consistent repayment history maintained", -1.45),
    ("strong liquidity reserve available", -1.35),
    ("stable revenue trajectory verified", -1.25),
    ("proactive early settlement completed", -1.40),
    ("zero dispute record sustained", -1.50),
]

_NEUTRAL_CLAUSES: List[str] = [
    "quarterly account review completed",
    "standard telemetry profile logged",
    "routine documentation check passed",
    "scheduled portfolio assessment updated",
]


def setup_logger(level: int = logging.INFO) -> None:
    """Configure root logging cleanly without duplicating handlers."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
            level=level,
        )
    else:
        root.setLevel(level)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TextTokenizer:
    """Self-contained deterministic word tokenizer with special [PAD], [UNK], and [CLS] tokens."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    CLS_TOKEN = "<cls>"
    PAD_ID = 0
    UNK_ID = 1
    CLS_ID = 2

    def __init__(self, vocab: Optional[Mapping[str, int]] = None) -> None:
        if vocab is None:
            self.token_to_id: Dict[str, int] = {
                self.PAD_TOKEN: self.PAD_ID,
                self.UNK_TOKEN: self.UNK_ID,
                self.CLS_TOKEN: self.CLS_ID,
            }
        else:
            self.token_to_id = {str(k): int(v) for k, v in vocab.items()}
        self.id_to_token: Dict[int, str] = {idx: tok for tok, idx in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Split lowercased text into deterministic alphanumeric tokens."""
        return _TOKEN_PATTERN.findall(text.lower())

    def fit(self, texts: Sequence[str], min_freq: int = 1) -> "TextTokenizer":
        """Build vocabulary deterministically from a training corpus."""
        counts: Dict[str, int] = {}
        for text in texts:
            for tok in self.tokenize(text):
                counts[tok] = counts.get(tok, 0) + 1

        self.token_to_id = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID,
            self.CLS_TOKEN: self.CLS_ID,
        }
        sorted_tokens = sorted(
            (tok for tok, freq in counts.items() if freq >= min_freq),
            key=lambda tok: (-counts[tok], tok),
        )
        for tok in sorted_tokens:
            if tok not in self.token_to_id:
                self.token_to_id[tok] = len(self.token_to_id)

        self.id_to_token = {idx: tok for tok, idx in self.token_to_id.items()}
        return self

    def encode(self, text: str, max_len: int = 32) -> List[int]:
        """Encode one text string into fixed-length token IDs starting with <cls>."""
        if max_len < 2:
            raise ValueError(f"max_len must be >= 2, got {max_len}")
        tokens = self.tokenize(text)[: max_len - 1]
        ids = [self.CLS_ID] + [self.token_to_id.get(tok, self.UNK_ID) for tok in tokens]
        if len(ids) < max_len:
            ids.extend([self.PAD_ID] * (max_len - len(ids)))
        return ids

    def encode_batch(self, texts: Sequence[str], max_len: int = 32) -> torch.Tensor:
        """Encode a sequence of texts into an int64 tensor of shape ``(B, max_len)``."""
        encoded = [self.encode(text, max_len=max_len) for text in texts]
        return torch.tensor(encoded, dtype=torch.long)

    def to_dict(self) -> Dict[str, int]:
        """Export vocabulary mapping as primitive dict for safe serialization."""
        return dict(self.token_to_id)

    @classmethod
    def from_dict(cls, vocab: Mapping[str, int]) -> "TextTokenizer":
        """Reconstruct tokenizer from a serialized vocabulary mapping."""
        return cls(vocab=vocab)


class MultimodalDataset(Dataset):
    """PyTorch Dataset pairing standardized tabular vectors, token IDs, and labels."""

    def __init__(
        self,
        X_tabular: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        if not (len(X_tabular) == len(input_ids) == len(labels)):
            raise ValueError("Tabular, text, and label tensors must have equal length")
        self.X_tabular = X_tabular
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_tabular[idx], self.input_ids[idx], self.labels[idx]


@dataclass(frozen=True)
class TabularStandardizer:
    """Simple tensor-friendly feature standardizer compatible with weights_only checkpoints."""

    mean: List[float]
    scale: List[float]

    @classmethod
    def fit(cls, X: np.ndarray) -> "TabularStandardizer":
        mean = X.mean(axis=0).astype(np.float64)
        scale = X.std(axis=0).astype(np.float64)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean=mean.tolist(), scale=scale.tolist())

    def transform(self, X: np.ndarray) -> torch.Tensor:
        mean_arr = np.asarray(self.mean, dtype=np.float32)
        scale_arr = np.asarray(self.scale, dtype=np.float32)
        scaled = (X.astype(np.float32) - mean_arr) / scale_arr
        return torch.from_numpy(scaled)


def generate_synthetic_data(
    num_samples: int = 1200,
    num_features: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Generate synthetic tabular+text pairs where BOTH modalities jointly determine the label.

    The target combines continuous tabular signals, semantic text risk/mitigation clauses,
    and a cross-modal interaction term so neither modality alone solves the task cleanly.
    """
    if num_features < 4:
        raise ValueError(f"num_features must be >= 4, got {num_features}")

    rng = np.random.default_rng(seed)
    X_tabular = rng.normal(0.0, 1.0, size=(num_samples, num_features)).astype(np.float32)

    # Introduce realistic feature correlations for auxiliary dimensions.
    if num_features > 4:
        X_tabular[:, 4] = 0.6 * X_tabular[:, 0] + 0.8 * X_tabular[:, 4]
    if num_features > 5:
        X_tabular[:, 5] = -0.5 * X_tabular[:, 1] + 0.85 * X_tabular[:, 5]

    tabular_signal = (
        1.15 * X_tabular[:, 0]
        - 0.95 * X_tabular[:, 1]
        + 0.65 * (X_tabular[:, 2] * X_tabular[:, 3])
        - 0.35 * X_tabular[:, 4]
    )

    texts: List[str] = []
    text_scores = np.zeros(num_samples, dtype=np.float64)

    for i in range(num_samples):
        regime = rng.choice(["adverse", "mixed", "mitigated"], p=[0.36, 0.28, 0.36])
        neutral = _NEUTRAL_CLAUSES[int(rng.integers(0, len(_NEUTRAL_CLAUSES)))]

        if regime == "adverse":
            c1, s1 = _ADVERSE_CLAUSES[int(rng.integers(0, len(_ADVERSE_CLAUSES)))]
            c2, s2 = _MODERATE_CLAUSES[int(rng.integers(0, len(_MODERATE_CLAUSES)))]
            clauses = [c1, c2, neutral]
            score = 0.75 * s1 + 0.45 * s2
        elif regime == "mitigated":
            c1, s1 = _MITIGATING_CLAUSES[int(rng.integers(0, len(_MITIGATING_CLAUSES)))]
            c2, s2 = _MITIGATING_CLAUSES[int(rng.integers(0, len(_MITIGATING_CLAUSES)))]
            clauses = [c1, c2, neutral]
            score = 0.65 * s1 + 0.45 * s2
        else:
            c1, s1 = _MODERATE_CLAUSES[int(rng.integers(0, len(_MODERATE_CLAUSES)))]
            c2, s2 = _MITIGATING_CLAUSES[int(rng.integers(0, len(_MITIGATING_CLAUSES)))]
            clauses = [c1, c2, neutral]
            score = 0.7 * s1 + 0.55 * s2

        rng.shuffle(clauses)
        texts.append(" ; ".join(clauses))
        text_scores[i] = score

    # Joint multimodal score with cross-modal interaction term.
    joint_logit = (
        1.25 * tabular_signal
        + 1.35 * text_scores
        + 0.75 * (tabular_signal * np.tanh(text_scores))
        + rng.normal(0.0, 0.25, size=num_samples)
    )
    y = (joint_logit > 0.0).astype(np.int64)
    return X_tabular, texts, y


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute Accuracy, Macro Precision, Macro Recall, Macro-F1, and ROC-AUC."""
    y_pred = y_prob.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        metrics["roc_auc"] = 0.5
    elif y_prob.shape[1] == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
    return metrics
