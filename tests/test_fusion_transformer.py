"""Comprehensive test suite for fusion-transformer."""

from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

from config import Config
from main import main as cli_main
from src.models import (
    FeatureTokenizer,
    FusionTransformer,
    SinusoidalPositionalEncoding,
    TextTransformerEncoder,
)
from src.pipeline import Pipeline
from src.utils import (
    MultimodalDataset,
    TabularStandardizer,
    TextTokenizer,
    compute_classification_metrics,
    generate_synthetic_data,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_heads": 0}, "n_heads must be >= 1"),
        ({"n_tabular_layers": 0}, "n_tabular_layers must be >= 1"),
        ({"n_text_layers": -1}, "n_text_layers must be >= 1"),
        ({"dropout": 1.0}, "dropout must be in [0, 1)"),
        ({"dropout": -0.1}, "dropout must be in [0, 1)"),
        ({"learning_rate": 0.0}, "learning_rate must be > 0"),
        ({"learning_rate": -1e-3}, "learning_rate must be > 0"),
        ({"learning_rate": float("nan")}, "learning_rate must be finite"),
        ({"learning_rate": float("inf")}, "learning_rate must be finite"),
        ({"weight_decay": -0.1}, "weight_decay must be >= 0"),
        ({"weight_decay": float("nan")}, "weight_decay must be finite"),
        ({"val_size": float("inf")}, "val_size must be finite"),
    ],
)
def test_config_rejects_invalid_model_hyperparameters(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        Config(**overrides)  # type: ignore[arg-type]


def test_forward_shapes_attention_and_no_nans() -> None:
    """FeatureTokenizer, TextTransformerEncoder, and CrossAttentionFusion produce valid finite tensors."""
    batch_size = 6
    num_features = 10
    seq_len = 16
    d_model = 32

    feat_tok = FeatureTokenizer(num_features=num_features, d_model=d_model)
    x_tab = torch.randn(batch_size, num_features)
    tab_tokens = feat_tok(x_tab)
    assert tab_tokens.shape == (batch_size, 1 + num_features, d_model)
    assert torch.isfinite(tab_tokens).all()

    tokenizer = TextTokenizer().fit(
        ["escalated compliance flag active", "verified collateral coverage confirmed"]
    )
    input_ids = tokenizer.encode_batch(
        ["escalated compliance flag", "verified collateral coverage confirmed extra unknown"],
        max_len=seq_len,
    )
    # Pad batch out to batch_size
    input_ids = input_ids.repeat(3, 1)

    model = FusionTransformer(
        num_features=num_features,
        vocab_size=tokenizer.vocab_size,
        num_classes=2,
        d_model=d_model,
        n_heads=4,
        n_tabular_layers=1,
        n_text_layers=1,
        dim_feedforward=64,
        max_seq_len=seq_len,
        dropout=0.0,
    )
    model.eval()
    logits, attn_maps = model(x_tab, input_ids, return_attention=True)

    assert logits.shape == (batch_size, 2)
    assert torch.isfinite(logits).all()
    assert attn_maps["tab_to_txt"].shape == (batch_size, 1 + num_features, seq_len)
    assert attn_maps["txt_to_tab"].shape == (batch_size, seq_len, 1 + num_features)
    assert torch.isfinite(attn_maps["tab_to_txt"]).all()
    assert torch.isfinite(attn_maps["txt_to_tab"]).all()


@pytest.mark.parametrize(
    "input_ids",
    [
        torch.tensor([[TextTokenizer.CLS_ID, -1]], dtype=torch.long),
        torch.tensor([[TextTokenizer.CLS_ID, 8]], dtype=torch.long),
        torch.tensor([[TextTokenizer.CLS_ID, 3.0]], dtype=torch.float32),
    ],
)
def test_text_encoder_rejects_invalid_token_ids(input_ids: torch.Tensor) -> None:
    encoder = TextTransformerEncoder(
        vocab_size=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dim_feedforward=32,
        max_seq_len=4,
        dropout=0.0,
    )

    with pytest.raises(ValueError, match="input_ids must contain integer token IDs in "):
        encoder(input_ids)


def test_text_encoder_rejects_sequences_beyond_positional_encoding_limit() -> None:
    encoder = TextTransformerEncoder(
        vocab_size=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dim_feedforward=32,
        max_seq_len=4,
        dropout=0.0,
    )

    with pytest.raises(ValueError, match="sequence length 5 exceeds max_seq_len 4"):
        encoder(torch.tensor([[TextTokenizer.CLS_ID, 3, 4, 5, 6]]))


def test_text_encoder_rejects_all_padding_sequences() -> None:
    encoder = TextTransformerEncoder(
        vocab_size=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dim_feedforward=32,
        max_seq_len=4,
        dropout=0.0,
    )

    with pytest.raises(ValueError, match="at least one non-padding token"):
        encoder(torch.full((1, 4), TextTokenizer.PAD_ID, dtype=torch.long))


def test_gradients_flow_through_both_encoders() -> None:
    """Loss backpropagation delivers non-zero gradients to both tabular and text encoders."""
    torch.manual_seed(42)
    tokenizer = TextTokenizer().fit(
        [
            "repeated late notices reported",
            "consistent repayment history maintained",
            "quarterly account review completed",
        ]
    )
    texts = [
        "repeated late notices reported",
        "consistent repayment history maintained",
        "quarterly account review completed",
        "repeated late notices quarterly account review",
    ]
    input_ids = tokenizer.encode_batch(texts, max_len=16)
    x_tab = torch.randn(4, 10)
    labels = torch.tensor([1, 0, 0, 1], dtype=torch.long)

    model = FusionTransformer(
        num_features=10,
        vocab_size=tokenizer.vocab_size,
        num_classes=2,
        d_model=32,
        n_heads=4,
        n_tabular_layers=2,
        n_text_layers=2,
        dim_feedforward=64,
        max_seq_len=16,
        dropout=0.0,
    )
    model.train()
    logits = model(x_tab, input_ids)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    # Check FeatureTokenizer + TabularTransformerEncoder gradients
    assert model.tabular_encoder.feature_tokenizer.weight.grad is not None
    assert model.tabular_encoder.feature_tokenizer.weight.grad.abs().sum().item() > 0.0
    assert model.tabular_encoder.feature_tokenizer.cls_token.grad is not None
    assert model.tabular_encoder.feature_tokenizer.cls_token.grad.abs().sum().item() > 0.0
    tab_enc_grad = sum(
        p.grad.abs().sum().item()
        for p in model.tabular_encoder.encoder.parameters()
        if p.grad is not None
    )
    assert tab_enc_grad > 0.0

    # Check TextTransformerEncoder embedding + encoder gradients
    assert model.text_encoder.embedding.weight.grad is not None
    assert model.text_encoder.embedding.weight.grad.abs().sum().item() > 0.0
    txt_enc_grad = sum(
        p.grad.abs().sum().item()
        for p in model.text_encoder.encoder.parameters()
        if p.grad is not None
    )
    assert txt_enc_grad > 0.0

    # Check CrossAttentionFusion gradients
    fusion_grad = sum(
        p.grad.abs().sum().item() for p in model.fusion.parameters() if p.grad is not None
    )
    assert fusion_grad > 0.0


def test_training_reduces_loss_and_beats_random_chance() -> None:
    """End-to-end training reduces loss and outperforms random chance on synthetic multimodal data."""
    cfg = Config(
        d_model=32,
        n_heads=4,
        n_tabular_layers=1,
        n_text_layers=1,
        dim_feedforward=64,
        num_samples=400,
        batch_size=64,
        num_epochs=8,
        seed=42,
    )
    pipeline = Pipeline(cfg)
    summary = pipeline.train()

    assert summary.train_losses[-1] < summary.train_losses[0]
    assert summary.test_metrics["accuracy"] > 0.80
    assert summary.test_metrics["macro_f1"] > 0.78
    assert summary.test_metrics["roc_auc"] > 0.85


def test_checkpoint_save_load_weights_only_roundtrip(tmp_path: Path) -> None:
    """Safe weights_only=True checkpoint save/load reproduces predictions identically."""
    ckpt_file = tmp_path / "fusion_test.pt"
    cfg = Config(
        d_model=32,
        n_heads=4,
        n_tabular_layers=1,
        n_text_layers=1,
        dim_feedforward=64,
        num_samples=300,
        batch_size=64,
        num_epochs=4,
        checkpoint_path=str(ckpt_file),
        seed=7,
    )
    train_pipe = Pipeline(cfg)
    train_pipe.train(checkpoint_path=str(ckpt_file))

    X_tab, texts, _ = generate_synthetic_data(num_samples=20, num_features=cfg.num_features, seed=99)
    probs_before = train_pipe.predict(X_tab, texts)

    # Verify raw torch.load with weights_only=True succeeds without errors.
    raw_payload = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    assert set(raw_payload.keys()) >= {
        "model_state_dict",
        "config",
        "vocab",
        "scaler_mean",
        "scaler_scale",
    }

    eval_pipe = Pipeline()
    eval_pipe.load_checkpoint(str(ckpt_file))
    probs_after = eval_pipe.predict(X_tab, texts)

    assert np.allclose(probs_before, probs_after, atol=1e-6)


def test_tokenizer_synthetic_data_and_metrics() -> None:
    """Verify deterministic tokenizer behavior, balanced synthetic data, and sklearn metrics."""
    tok = TextTokenizer().fit(["alpha beta", "beta gamma"])
    ids = tok.encode("Alpha unknown beta gamma delta", max_len=6)
    assert ids[0] == TextTokenizer.CLS_ID
    assert ids[2] == TextTokenizer.UNK_ID
    assert len(ids) == 6

    X1, t1, y1 = generate_synthetic_data(num_samples=300, num_features=10, seed=42)
    X2, t2, y2 = generate_synthetic_data(num_samples=300, num_features=10, seed=42)
    assert np.array_equal(X1, X2)
    assert t1 == t2
    assert np.array_equal(y1, y2)
    assert 0.2 < y1.mean() < 0.8

    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    metrics = compute_classification_metrics(y_true, y_prob)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_cli_train_evaluate_and_demo(tmp_path: Path) -> None:
    """CLI --train, --evaluate, and --demo commands execute cleanly."""
    ckpt = tmp_path / "cli_model.pt"
    assert cli_main(["--train", "--epochs", "2", "--batch-size", "64", "--checkpoint", str(ckpt)]) == 0
    assert ckpt.exists()
    assert cli_main(["--evaluate", "--batch-size", "64", "--checkpoint", str(ckpt)]) == 0

    proc = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--demo",
            "--epochs",
            "1",
            "--batch-size",
            "64",
            "--checkpoint",
            str(tmp_path / "demo.pt"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Demo Complete:" in proc.stdout
    assert "Sample Multimodal Predictions:" in proc.stdout


def test_exit_code_no_threshold_always_succeeds() -> None:
    from main import exit_code_for

    assert exit_code_for({"macro_f1": 0.0}, None) == 0


def test_exit_code_gates_on_min_macro_f1() -> None:
    from main import exit_code_for

    assert exit_code_for({"macro_f1": 0.5}, 0.6) == 1
    assert exit_code_for({"macro_f1": 0.6}, 0.6) == 0
    assert exit_code_for({"macro_f1": 0.9}, 0.6) == 0


def test_exit_code_rejects_invalid_threshold() -> None:
    from main import exit_code_for

    for bad in (float("nan"), float("inf"), float("-inf"), -0.1, 1.5):
        with pytest.raises(ValueError, match="min_macro_f1 must be a finite value"):
            exit_code_for({"macro_f1": 1.0}, bad)


def test_cli_min_macro_f1_gates_the_run(tmp_path: Path) -> None:
    argv = ["--train", "--epochs", "2", "--batch-size", "64", "--checkpoint", str(tmp_path / "g.pt")]
    assert cli_main(argv + ["--min-macro-f1", "0.0"]) == 0
    assert cli_main(argv + ["--min-macro-f1", "1.0"]) == 1


class TestTabularStandardizer:
    @pytest.mark.parametrize(
        ("X", "message"),
        [
            (np.empty((0, 2)), "X must contain at least one row"),
            (np.empty((4, 0)), "X must contain at least one feature"),
            (np.array([1.0, 2.0]), "X must be a 2D array"),
            (np.array([[1.0, np.nan]]), "X must contain only finite values"),
        ],
    )
    def test_fit_rejects_invalid_input(self, X: np.ndarray, message: str) -> None:
        with pytest.raises(ValueError, match=re.escape(message)):
            TabularStandardizer.fit(X)

    def test_transform_rejects_feature_count_mismatch(self) -> None:
        standardizer = TabularStandardizer.fit(np.ones((3, 2)))
        with pytest.raises(ValueError, match="expected 2 features"):
            standardizer.transform(np.ones((4, 3)))

    def test_transform_rejects_non_finite_values(self) -> None:
        standardizer = TabularStandardizer.fit(np.ones((3, 2)))
        with pytest.raises(ValueError, match="X must contain only finite values"):
            standardizer.transform(np.array([[1.0, np.inf]]))

    def test_transformed_columns_have_zero_mean_unit_std(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(loc=5.0, scale=2.0, size=(200, 3))
        std = TabularStandardizer.fit(X)
        Z = std.transform(X).numpy()
        assert Z.mean(axis=0) == pytest.approx(np.zeros(3), abs=1e-4)
        assert Z.std(axis=0) == pytest.approx(np.ones(3), abs=1e-2)

    def test_constant_column_uses_unit_scale(self) -> None:
        X = np.full((10, 1), 7.0)
        std = TabularStandardizer.fit(X)
        assert std.scale == [1.0]
        assert std.transform(X).numpy() == pytest.approx(np.zeros((10, 1)))


class TestMultimodalDataset:
    def test_len_and_getitem_shapes(self) -> None:
        ds = MultimodalDataset(
            X_tabular=torch.zeros(5, 2),
            input_ids=torch.zeros(5, 4, dtype=torch.long),
            labels=torch.zeros(5, dtype=torch.long),
        )
        assert len(ds) == 5
        x_tab, ids, y = ds[0]
        assert x_tab.shape == (2,)
        assert ids.shape == (4,)
        assert y.shape == ()

    def test_mismatched_lengths_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            MultimodalDataset(
                X_tabular=torch.zeros(3, 2),
                input_ids=torch.zeros(4, 2, dtype=torch.long),
                labels=torch.zeros(3, dtype=torch.long),
            )


class TestSinusoidalPositionalEncoding:
    def test_output_matches_input_shape(self) -> None:
        pe = SinusoidalPositionalEncoding(d_model=8, max_len=16)
        x = torch.zeros(2, 5, 8)
        out = pe(x)
        assert out.shape == x.shape

    def test_positions_are_distinct(self) -> None:
        pe = SinusoidalPositionalEncoding(d_model=8, max_len=16)
        x = torch.zeros(1, 4, 8)
        out = pe(x)[0]
        assert not torch.allclose(out[0], out[1])


class TestSetSeed:
    def test_torch_stream_is_reproducible(self) -> None:
        set_seed(99)
        a = torch.randn(4).tolist()
        set_seed(99)
        assert torch.randn(4).tolist() == a
