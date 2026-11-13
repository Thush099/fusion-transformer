"""Training, evaluation, checkpoint serialization, and demo pipeline."""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from src.models import FusionTransformer
from src.utils import (
    MultimodalDataset,
    TabularStandardizer,
    TextTokenizer,
    compute_classification_metrics,
    generate_synthetic_data,
    set_seed,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingSummary:
    """History and held-out metrics recorded during training."""

    train_losses: List[float]
    val_losses: List[float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    checkpoint_path: str


class Pipeline:
    """Manages data preparation, FusionTransformer training, checkpointing, and evaluation."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: Optional[TextTokenizer] = None
        self.standardizer: Optional[TabularStandardizer] = None
        self.model: Optional[FusionTransformer] = None

    def _prepare_splits(
        self,
    ) -> Tuple[
        Tuple[np.ndarray, List[str], np.ndarray],
        Tuple[np.ndarray, List[str], np.ndarray],
        Tuple[np.ndarray, List[str], np.ndarray],
    ]:
        X_tab, X_text, y = generate_synthetic_data(
            num_samples=self.config.num_samples,
            num_features=self.config.num_features,
            seed=self.config.seed,
        )
        indices = np.arange(len(y))
        holdout_frac = self.config.val_size + self.config.test_size
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=holdout_frac,
            stratify=y,
            random_state=self.config.seed,
        )
        rel_test_frac = self.config.test_size / holdout_frac
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=rel_test_frac,
            stratify=y[temp_idx],
            random_state=self.config.seed,
        )

        def gather(idx: np.ndarray) -> Tuple[np.ndarray, List[str], np.ndarray]:
            return X_tab[idx], [X_text[int(i)] for i in idx], y[idx]

        return gather(train_idx), gather(val_idx), gather(test_idx)

    def _build_loader(
        self,
        X_tab_np: np.ndarray,
        texts: List[str],
        y_np: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        if self.tokenizer is None or self.standardizer is None:
            raise RuntimeError("Tokenizer and standardizer must be initialized first")
        x_tab_tensor = self.standardizer.transform(X_tab_np)
        input_ids = self.tokenizer.encode_batch(texts, max_len=self.config.max_seq_len)
        labels = torch.from_numpy(y_np.astype(np.int64))
        dataset = MultimodalDataset(x_tab_tensor, input_ids, labels)
        generator = torch.Generator().manual_seed(self.config.seed) if shuffle else None
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    def _build_model(self, vocab_size: int) -> FusionTransformer:
        model = FusionTransformer(
            num_features=self.config.num_features,
            vocab_size=vocab_size,
            num_classes=self.config.num_classes,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_tabular_layers=self.config.n_tabular_layers,
            n_text_layers=self.config.n_text_layers,
            dim_feedforward=self.config.dim_feedforward,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout,
            pad_token_id=TextTokenizer.PAD_ID,
        )
        return model.to(self.device)

    @torch.no_grad()
    def evaluate_loader(self, loader: DataLoader) -> Tuple[float, Dict[str, float], np.ndarray]:
        """Evaluate ``self.model`` over a DataLoader and return loss, metrics, and probabilities."""
        if self.model is None:
            raise RuntimeError("Model has not been built or loaded")
        self.model.eval()
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_examples = 0
        all_probs: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        for x_tab, input_ids, labels in loader:
            x_tab = x_tab.to(self.device)
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(x_tab, input_ids)
            loss = loss_fn(logits, labels)
            probs = torch.softmax(logits, dim=-1)

            batch_n = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_n
            total_examples += batch_n
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        y_prob = np.concatenate(all_probs, axis=0)
        y_true = np.concatenate(all_labels, axis=0)
        avg_loss = total_loss / max(total_examples, 1)
        metrics = compute_classification_metrics(y_true, y_prob)
        return avg_loss, metrics, y_prob

    def save_checkpoint(self, path: Optional[str] = None) -> Path:
        """Save model weights, config, vocabulary, and scaler statistics safely."""
        if self.model is None or self.tokenizer is None or self.standardizer is None:
            raise RuntimeError("Cannot save checkpoint before model, tokenizer, and scaler exist")
        target = Path(path or self.config.checkpoint_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "config": asdict(self.config),
            "vocab": self.tokenizer.to_dict(),
            "scaler_mean": list(self.standardizer.mean),
            "scaler_scale": list(self.standardizer.scale),
        }
        torch.save(payload, target)
        logger.info("Saved checkpoint to %s", target)
        return target

    def load_checkpoint(self, path: Optional[str] = None) -> Path:
        """Load checkpoint strictly via ``torch.load(..., weights_only=True)``."""
        target = Path(path or self.config.checkpoint_path)
        if not target.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {target}. Run with --train first."
            )
        payload = torch.load(target, map_location=self.device, weights_only=True)
        self.config = Config(**payload["config"])
        self.tokenizer = TextTokenizer.from_dict(payload["vocab"])
        self.standardizer = TabularStandardizer(
            mean=[float(x) for x in payload["scaler_mean"]],
            scale=[float(x) for x in payload["scaler_scale"]],
        )
        self.model = self._build_model(vocab_size=self.tokenizer.vocab_size)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        logger.info("Loaded checkpoint from %s", target)
        return target

    def train(self, checkpoint_path: Optional[str] = None) -> TrainingSummary:
        """Train FusionTransformer on synthetic multimodal splits and save the best checkpoint."""
        set_seed(self.config.seed)
        (X_tr, txt_tr, y_tr), (X_val, txt_val, y_val), (X_te, txt_te, y_te) = (
            self._prepare_splits()
        )

        self.standardizer = TabularStandardizer.fit(X_tr)
        self.tokenizer = TextTokenizer().fit(txt_tr)
        self.model = self._build_model(vocab_size=self.tokenizer.vocab_size)

        train_loader = self._build_loader(X_tr, txt_tr, y_tr, shuffle=True)
        val_loader = self._build_loader(X_val, txt_val, y_val, shuffle=False)
        test_loader = self._build_loader(X_te, txt_te, y_te, shuffle=False)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs,
        )
        loss_fn = nn.CrossEntropyLoss()

        train_losses: List[float] = []
        val_losses: List[float] = []
        best_val_f1 = -1.0
        best_state: Optional[Dict[str, torch.Tensor]] = None

        for epoch in range(1, self.config.num_epochs + 1):
            self.model.train()
            running_loss = 0.0
            seen = 0
            for x_tab, input_ids, labels in train_loader:
                x_tab = x_tab.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(x_tab, input_ids)
                loss = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                batch_n = int(labels.shape[0])
                running_loss += float(loss.item()) * batch_n
                seen += batch_n

            scheduler.step()
            train_loss = running_loss / max(seen, 1)
            val_loss, val_metrics, _ = self.evaluate_loader(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_metrics["macro_f1"] >= best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }

            logger.info(
                "Epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f | "
                "val_acc=%.4f val_f1=%.4f val_auc=%.4f",
                epoch,
                self.config.num_epochs,
                train_loss,
                val_loss,
                val_metrics["accuracy"],
                val_metrics["macro_f1"],
                val_metrics["roc_auc"],
            )

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        _, final_val_metrics, _ = self.evaluate_loader(val_loader)
        _, test_metrics, _ = self.evaluate_loader(test_loader)
        saved_path = self.save_checkpoint(checkpoint_path)

        logger.info(
            "Test Metrics | acc=%.4f prec=%.4f rec=%.4f macro_f1=%.4f roc_auc=%.4f",
            test_metrics["accuracy"],
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["macro_f1"],
            test_metrics["roc_auc"],
        )
        return TrainingSummary(
            train_losses=train_losses,
            val_losses=val_losses,
            val_metrics=final_val_metrics,
            test_metrics=test_metrics,
            checkpoint_path=str(saved_path),
        )

    def evaluate(self, checkpoint_path: Optional[str] = None) -> Dict[str, float]:
        """Load a saved checkpoint and evaluate on the held-out test split."""
        self.load_checkpoint(checkpoint_path)
        _, _, (X_te, txt_te, y_te) = self._prepare_splits()
        test_loader = self._build_loader(X_te, txt_te, y_te, shuffle=False)
        test_loss, metrics, _ = self.evaluate_loader(test_loader)
        metrics["loss"] = test_loss
        logger.info(
            "Evaluation Results | loss=%.4f acc=%.4f prec=%.4f rec=%.4f macro_f1=%.4f roc_auc=%.4f",
            test_loss,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["macro_f1"],
            metrics["roc_auc"],
        )
        return metrics

    @torch.no_grad()
    def predict(self, X_tabular: np.ndarray, texts: List[str]) -> np.ndarray:
        """Run inference and return class probabilities of shape ``(N, num_classes)``."""
        if self.model is None or self.tokenizer is None or self.standardizer is None:
            raise RuntimeError("Model, tokenizer, and standardizer must be loaded first")
        self.model.eval()
        x_tab_t = self.standardizer.transform(X_tabular).to(self.device)
        input_ids = self.tokenizer.encode_batch(texts, max_len=self.config.max_seq_len).to(
            self.device
        )
        logits = self.model(x_tab_t, input_ids)
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def demo(self, num_examples: int = 3) -> Dict[str, float]:
        """Train, evaluate, and print sample multimodal predictions end-to-end."""
        summary = self.train()
        _, _, (X_te, txt_te, y_te) = self._prepare_splits()
        sample_X = X_te[:num_examples]
        sample_txt = txt_te[:num_examples]
        sample_y = y_te[:num_examples]
        probs = self.predict(sample_X, sample_txt)

        print("\nSample Multimodal Predictions:")
        for idx in range(len(sample_y)):
            pred_cls = int(probs[idx].argmax())
            conf = float(probs[idx, pred_cls])
            print(
                f"  [{idx + 1}] true={int(sample_y[idx])} pred={pred_cls} "
                f"p(high_risk)={probs[idx, 1]:.3f} conf={conf:.3f} | "
                f"tab[0:2]=[{sample_X[idx, 0]:+.2f}, {sample_X[idx, 1]:+.2f}] | "
                f"text='{sample_txt[idx]}'"
            )
        return summary.test_metrics
