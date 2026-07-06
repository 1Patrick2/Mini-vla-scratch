# Training Pipeline

Stage 3 implements behavior cloning (BC) training loop for MiniVLA:

```
toy_2d episode data → Toy2DDataset → DataLoader
→ MiniVLA forward → MSE loss → backward → optimizer.step
→ checkpoint save/load → CLI
```

**Loss:** MSE(action_pred, action_gt)

**Frozen by default:** text_encoder

**Trainable:** vision_encoder, state_encoder, fusion, action_head

**Key files:**
- `mini_vla/training/losses.py` — MSE action loss
- `mini_vla/training/metrics.py` — MSE / L1 evaluation metrics
- `mini_vla/training/optimizer.py` — Adam optimizer (skips frozen params)
- `mini_vla/training/trainer.py` — Trainer class (fit / train_one_epoch)
- `mini_vla/training/checkpoint.py` — Save/load last.pt and best.pt
- `scripts/train.py` — Training CLI entry point

**Validation:**
```bash
pytest --tb=short
python scripts/train.py --config configs/train/debug.yaml
```

Stage 4 will add inference and visualization. See `docs/09_training_design.md` for the full training design document.
