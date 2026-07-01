# Model Architecture

## Stage 2 Current Model (Toy 2D Mini VLA)

### Forward Pass

```
image [B, 3, 64, 64]
  → SmallCNNVisionEncoder
  → image_feat [B, 128]

input_ids [B, T] + attention_mask [B, T]
  → LLM-ready TextBackbone
  → text_feat [B, 128]

state [B, 2]
  → StateEncoder (MLP)
  → state_feat [B, 128]

concat(image_feat, text_feat, state_feat) [B, 384]
  → FusionMLP
  → fused_feat [B, 128]

fused_feat
  → ActionHead (continuous MLP regression)
  → action_pred [B, action_dim]
```

### Text Backbone Design

The text encoder uses an LLM-ready abstraction layer (`BaseTextEncoder`):

| Implementation | Purpose |
|----------------|---------|
| `MockLLMTextEncoder` | Default for Stage 2 testing — Embedding + mask-aware mean pooling + Linear projection. No `transformers` dependency. |
| `LLMTextEncoder` | Optional HuggingFace / local LLM adapter. Raises a clear error if `transformers` is not installed. |

**Key constraints:**
- `pytest` does NOT require downloading any real LLM or HuggingFace models.
- Stage 3 training will **freeze** the text backbone by default.

### Action Representation

- **Continuous MLP regression** — the model outputs raw action deltas `[dx, dy]`.
- No action tokenization.
- No diffusion.
- No flow matching.
- No action chunking (reserved for future versions).

### Encoder Details

#### SmallCNNVisionEncoder

| Layer | Output Shape |
|-------|-------------|
| Conv2d(3, 16, 3, padding=1) + ReLU + MaxPool2d(2) | [B, 16, 32, 32] |
| Conv2d(16, 32, 3, padding=1) + ReLU + MaxPool2d(2) | [B, 32, 16, 16] |
| Conv2d(32, 64, 3, padding=1) + ReLU + AdaptiveAvgPool2d(1) | [B, 64, 1, 1] |
| Flatten + Linear(64, output_dim) | [B, output_dim] |

#### MockLLMTextEncoder

| Layer | Output Shape |
|-------|-------------|
| Embedding(vocab_size, hidden_dim, padding_idx=0) | [B, T, hidden_dim] |
| Mask-aware mean pooling | [B, hidden_dim] |
| Linear(hidden_dim, output_dim) | [B, output_dim] |

#### StateEncoder

```
Linear(2, 64) → ReLU → Linear(64, output_dim) → [B, output_dim]
```

#### FusionMLP

```
Concat(image_feat, text_feat, state_feat) [B, 384]
  → Linear(384, 256) → ReLU → Linear(256, 128) → ReLU → [B, 128]
```

#### ActionHead

```
Linear(128, 128) → ReLU → Linear(128, action_dim) → [B, action_dim]
```

## References

- TinyVLA: compact VLA, fast inference, data efficiency. ([arXiv 2409.12514](https://arxiv.org/abs/2409.12514))
- SmolVLA: affordable VLA, LLM/VLM-ready text backbone. ([arXiv 2506.01844](https://arxiv.org/abs/2506.01844))
