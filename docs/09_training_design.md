# Stage 3 Training Design

## Goal

Implement a minimal Behavior Cloning (BC) training loop for MiniVLA:

```
toy_2d episode data → Toy2DDataset → DataLoader
→ MiniVLA forward → MSE loss → backward → optimizer.step
→ checkpoint save/load → CLI integration
```

## Loss Function

Use plain MSE (Mean Squared Error) between predicted action and ground-truth expert action:

```
loss = MSE(action_pred, action_gt)
```

- `action_pred`: Tensor[B, action_dim] — model output
- `action_gt`: Tensor[B, action_dim] — expert action from dataset
- Output: scalar Tensor, suitable for `backward()`

## Frozen vs Trainable Modules

By default during Stage 3:

| Module | Frozen? | Reason |
|--------|---------|--------|
| `vision_encoder` (SmallCNN) | ❌ Trainable | Learn visual features from toy data |
| `text_encoder` (MockLLMTextEncoder) | ✅ **Frozen** | Simulates frozen LLM backbone |
| `state_encoder` (MLP) | ❌ Trainable | Learn state features |
| `fusion` (FusionMLP) | ❌ Trainable | Learn cross-modal fusion |
| `action_head` (MLP) | ❌ Trainable | Learn action regression |

Freezing is controlled via `text_encoder.freeze: true` in config. When enabled, `build_model()` sets `requires_grad = False` on all text_encoder parameters, and `create_optimizer()` only collects parameters where `requires_grad == True`.

## Config Organization

Training pipeline uses two config files:

| Config | Path | Purpose |
|--------|------|---------|
| Model config | `configs/model/mini_vla_cnn_llm.yaml` | MiniVLA architecture (vision/text/state/fusion/action) |
| Training config | `configs/train/debug.yaml` | Data, optimizer, device, output paths |

The training config inherits from `base.yaml` via the `base:` field and adds its own `data.train` and `model_config` keys.

```
# configs/train/debug.yaml
base: ../base.yaml
model_config: configs/model/mini_vla_cnn_llm.yaml  # optional reference

data:
  batch_size: 4

train:
  epochs: 1
  lr: 0.001
  device: cpu
  num_workers: 0
```

## Checkpoint Strategy

- **Save frequency**: end of each epoch
- **Last checkpoint**: `outputs/checkpoints/last.pt` — always saved
- **Best checkpoint**: `outputs/checkpoints/best.pt` — saved when validation loss improves
- **Saved contents**:
  ```python
  {
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "epoch": epoch,
      "metrics": {"train_loss": ..., "train_mae": ...},
      "config": config,
  }
  ```

## Out of Scope (Not Doing Now)

The following features are deferred or out of scope for Stage 3:

| Feature | Reason |
|---------|--------|
| LoRA | Would complicate frozen text backbone; not needed for toy data |
| Real LLM finetuning | MockLLMTextEncoder has no real LLM weights |
| LeRobot dataset | Toy2D data is sufficient for training loop validation |
| LIBERO benchmark | External dependency; deferred to V1+ |
| WandB logging | Overhead for minimal pipeline; console print is sufficient |
| Multi-GPU / Distributed | Single-device CPU/GPU training only |
| Diffusion / Flow Matching | Action is continuous MLP regression, not diffusion |
| Action chunking | Single-step action prediction only |
| Real robot rollout | Deferred to Stage 5 |

## Training Loop Pseudocode

```python
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    epoch_mae = 0.0

    for batch in train_loader:
        action_pred = model(batch)                  # forward
        loss = mse_action_loss(action_pred, batch["action"])
        loss.backward()                              # backward
        optimizer.step()                             # update
        optimizer.zero_grad()

        epoch_loss += loss.item()
        epoch_mae += action_mae(action_pred, batch["action"])

    avg_loss = epoch_loss / len(train_loader)
    avg_mae = epoch_mae / len(train_loader)
    print(f"epoch={epoch} train_loss={avg_loss:.4f} train_mae={avg_mae:.4f}")

    save_checkpoint(...)
```

## Verification

```bash
# All tests pass
pytest --tb=short

# Single training step end-to-end
pytest tests/test_training_step.py -v -k "end_to_end"

# Full training run
python scripts/train.py --config configs/train/debug.yaml
```
