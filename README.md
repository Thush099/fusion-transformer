# fusion-transformer: Multimodal Tabular + Text Fusion Transformer

A self-contained PyTorch multimodal classification model combining **FT-Transformer** tabular feature tokenization, a deterministic **Transformer text encoder**, and **bidirectional cross-attention** with gated fusion.

## Architecture

```
  Tabular Features (B, F)               Raw Text Batch (List[str])
           |                                       |
           v                                       v
   FeatureTokenizer                         TextTokenizer (<cls>, <pad>, <unk>)
   + [CLS] Token (B, F+1, D)                + Embedding & Positional Encoding
           |                                       |
           v                                       v
  TabularTransformerEncoder               TextTransformerEncoder
  (nn.TransformerEncoder)                 (nn.TransformerEncoder w/ pad mask)
           |                                       |
           +-------------------+-------------------+
                               |
                               v
                      CrossAttentionFusion
             - Tabular <-> Text nn.MultiheadAttention
             - Residual LayerNorm + Gated [CLS] Projection
             - Bilinear Interaction + MLP Classification Head
                               |
                               v
                      Class Logits (B, C)
```

### Key Components

- **`FeatureTokenizer` & `TabularTransformerEncoder`** ([src/models.py](src/models.py)): Implements FT-Transformer style per-feature affine embeddings (`x_j * W_j + b_j`) plus a prepended learnable `[CLS]` token processed through `nn.TransformerEncoder`.
- **`TextTokenizer` & `TextTransformerEncoder`** ([src/utils.py](src/utils.py), [src/models.py](src/models.py)): Self-contained deterministic word tokenizer (no external pretrained weights required), sinusoidal positional encoding, padding masks, and `nn.TransformerEncoder`.
- **`CrossAttentionFusion`** ([src/models.py](src/models.py)): Uses `nn.MultiheadAttention(..., batch_first=True)` for tabular-to-text and text-to-tabular attention, residual `LayerNorm`, gated modality projection, and interaction MLP head.
- **Multimodal Synthetic Dataset** ([src/utils.py](src/utils.py)): Generates continuous tabular features paired with qualitative financial/risk text clauses where the label depends on both modalities and their cross-modal interaction term.
- **Safe Checkpointing & Metrics** ([src/pipeline.py](src/pipeline.py)): Saves and loads checkpoints via `torch.load(..., weights_only=True)` (primitive config/vocab/scaler state + tensor `state_dict`), and reports **Accuracy**, **Macro Precision**, **Macro Recall**, **Macro-F1**, and **ROC-AUC**. The tabular standardizer rejects empty, non-finite, or mismatched feature arrays before producing invalid scaler state or model inputs.

## Repository Layout

- `config.py` — Validated `Config` dataclass for model, training, and split settings; rejects invalid dimensions, non-finite values, and out-of-range `max_seq_len`, `num_samples`, and `seed` at construction.
- `main.py` — CLI entry point supporting `--train`, `--evaluate`, and `--demo`.
- `src/models.py` — `FeatureTokenizer`, `TabularTransformerEncoder`, `TextTransformerEncoder`, `CrossAttentionFusion`, and `FusionTransformer`. Every module validates its dimension arguments at construction (positive integer sizes, `d_model` divisible by `n_heads`, `dropout` in `[0, 1)`, `num_classes >= 2`, and `pad_token_id` within the vocabulary), so direct construction fails with a clear `ValueError` instead of a degenerate module or an opaque framework error.
- `src/utils.py` — `TextTokenizer`, `TabularStandardizer`, `MultimodalDataset`, synthetic multimodal generator, and `scikit-learn` metric helpers.
- `src/pipeline.py` — `Pipeline` handling stratified train/val/test splits, `DataLoader` training, checkpointing, evaluation, and inference.
- `tests/test_fusion_transformer.py` — Unit and integration tests for shapes, gradient flow, training convergence, checkpoint roundtrip, and CLI execution.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train the model and save the best checkpoint (`outputs/fusion_transformer.pt`):

```bash
python main.py --train
```

Add `--min-macro-f1 FLOAT` to gate a CI job: the run exits non-zero when its macro
F1 falls below the threshold in `[0, 1]`, and an out-of-range threshold is rejected
rather than silently disabling the gate:

```bash
python main.py --train --min-macro-f1 0.75
```

Evaluate a saved checkpoint on the held-out test split:

```bash
python main.py --evaluate
```

Run an end-to-end train, evaluation, and sample inference demo (also the default when no flags are provided):

```bash
python main.py --demo
```

### Training and evaluation flags

`--train` and `--evaluate` accept the following options:

| Flag | Default | Meaning |
|---|---:|---|
| `--epochs` | `15` | Number of training epochs |
| `--batch-size` | `32` | Mini-batch size |
| `--lr` | `2e-3` | AdamW learning rate (must be finite and `> 0`) |
| `--checkpoint` | `outputs/fusion_transformer.pt` | Checkpoint path used for saving and loading |
| `--seed` | `42` | Random seed |

```bash
python main.py --train --epochs 30 --batch-size 64 --lr 1e-3 --checkpoint outputs/run2.pt
python main.py --evaluate --checkpoint outputs/run2.pt
```

### Observed CLI Train & Evaluate Output

```text
$ python main.py --train
12:33:32 | INFO | Epoch 01/15 | train_loss=0.5608 | val_loss=0.4631 | val_acc=0.8389 val_f1=0.8111 val_auc=0.8915
...
12:34:02 | INFO | Epoch 12/15 | train_loss=0.1571 | val_loss=0.2801 | val_acc=0.9167 val_f1=0.9081 val_auc=0.9668
...
12:34:10 | INFO | Epoch 15/15 | train_loss=0.1049 | val_loss=0.2602 | val_acc=0.9056 val_f1=0.8973 val_auc=0.9714
12:34:10 | INFO | Saved checkpoint to outputs/fusion_transformer.pt
12:34:10 | INFO | Test Metrics | acc=0.9056 prec=0.9121 rec=0.8840 macro_f1=0.8950 roc_auc=0.9685

Training Complete (checkpoint=outputs/fusion_transformer.pt): test_acc=0.9056, precision=0.9121, recall=0.8840, macro_f1=0.8950, roc_auc=0.9685

$ python main.py --evaluate
12:35:09 | INFO | Loaded checkpoint from outputs/fusion_transformer.pt
12:35:11 | INFO | Evaluation Results | loss=0.2680 acc=0.9056 prec=0.9121 rec=0.8840 macro_f1=0.8950 roc_auc=0.9685

Evaluation Complete: loss=0.2680, accuracy=0.9056, precision=0.9121, recall=0.8840, macro_f1=0.8950, roc_auc=0.9685
```

## Running Tests

```bash
pytest -v
```

## License

MIT License — see [LICENSE](LICENSE).

## Web UI

A local training console drives the real pipeline in the browser:

```bash
python ui/server.py            # http://127.0.0.1:8130
```

Configure the multimodal run (epochs, samples, features, model dim), train on synthetic
tabular+text data, and read the modality-fusion diagram, training/validation loss curves,
and held-out accuracy, precision, recall, macro-F1, and ROC-AUC. The server uses only the
standard library and calls `Pipeline.train()`.
