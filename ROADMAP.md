# Roadmap

## Stage 0: Project Skeleton

**Goal:** Set up project structure, config system, environment files, and basic tests.

| Component | Status |
|-----------|--------|
| Root files (README, SETUP, PRINCIPLES, etc.) | ✅ Complete |
| pyproject.toml with dev tool config | ✅ Complete |
| Config system (loader + schema) | ✅ Complete |
| Training CLI dry-run | ✅ Complete |
| Action clipping utility | ✅ Complete |
| Safety wrapper | ⏳ Placeholder, planned for later stage |
| Basic unit tests (4 passing) | ✅ Complete |
| Environment scripts (conda/venv/WSL/Windows) | ✅ Complete |

**Command:** `pytest` — 4 passed
**Command:** `python scripts/train.py --config configs/train/debug.yaml --dry-run`

---

## Stage 1: Toy 2D Data Generation and Dataset

**Goal:** Generate synthetic 2D manipulation episodes and implement the dataset loader.

**Main files:**
- `scripts/generate_toy_data.py`
- `mini_vla/datasets/toy_2d_dataset.py`
- `mini_vla/datasets/collate.py`
- `mini_vla/datasets/transforms.py`
- `tests/test_dataset.py`

**Data format (per-episode):**
```
data/toy_2d/
└── episodes/
    ├── ep_000000/
    │   ├── frames/
    │   │   ├── 000000.png
    │   │   └── ...
    │   └── episode.json
    └── ep_000001/
        ├── frames/
        └── episode.json
```

Each `episode.json` contains per-step samples:
```json
{
  "episode_id": "ep_000000",
  "instruction": "move red object to target",
  "steps": [
    {
      "frame": "frames/000000.png",
      "state": [0.25, 0.40],
      "action": [0.03, -0.01],
      "target": [0.80, 0.20]
    }
  ]
}
```

**Command:**
```bash
python scripts/generate_toy_data.py --config configs/data/toy_2d.yaml --num-episodes 100
python -c "from mini_vla.datasets.toy_2d_dataset import Toy2DDataset; ds = Toy2DDataset('data/toy_2d'); print(ds[0].keys())"
```

**Acceptance:** `pytest tests/test_dataset.py` — dataset returns image/input_ids/state/action tensors with correct shapes.

---

## Stage 2: MiniVLA Model Forward

**Goal:** Implement `image + instruction + state → action` forward pass.

**Architecture:**
```
image → CNN vision_encoder → image_feat
text → Embedding + GRU language_encoder → text_feat
state → MLP state_encoder → state_feat
concat → Fusion MLP → ActionHead MLP → action_pred
```

**Main files:**
- `mini_vla/models/vision_encoder.py`
- `mini_vla/models/language_encoder.py`
- `mini_vla/models/state_encoder.py`
- `mini_vla/models/fusion.py`
- `mini_vla/models/action_head.py`
- `mini_vla/models/mini_vla.py`
- `tests/test_model_forward.py`

**Command:** `pytest tests/test_model_forward.py -v`

**Acceptance:** Model forward outputs `(batch, action_dim)` tensor.

---

## Stage 3: Behavior Cloning Training Loop

**Goal:** Run end-to-end training: dataset → model → loss → backward → checkpoint.

**Main files:**
- `mini_vla/training/losses.py`
- `mini_vla/training/metrics.py`
- `mini_vla/training/optimizer.py`
- `mini_vla/training/checkpoint.py`
- `mini_vla/training/trainer.py` (full implementation)
- `tests/test_training_step.py`

**Command:**
```bash
python scripts/train.py --config configs/train/debug.yaml
```

**Acceptance:** Training loss decreases over epochs, checkpoint saved to `outputs/checkpoints/best.pt`.

---

## Stage 4: Inference and Visualization

**Goal:** Load a trained model, predict action for a single sample, and visualize.

**Main files:**
- `mini_vla/inference/predictor.py`
- `mini_vla/inference/visualizer.py`
- `scripts/infer_one.py`

**Command:**
```bash
python scripts/infer_one.py --ckpt outputs/checkpoints/best.pt --config configs/train/debug.yaml
```

**Acceptance:** Generates `outputs/predictions/prediction.png` with direction arrow overlay.

---

## Stage 5: Fake Robot Rollout

**Goal:** Continuous action prediction loop with fake robot state update and success evaluation.

**Main files:**
- `mini_vla/inference/rollout.py`
- `mini_vla/robot_interface/fake_robot.py`
- `scripts/rollout_fake_robot.py`

**Command:**
```bash
python scripts/rollout_fake_robot.py --ckpt outputs/checkpoints/best.pt --num-episodes 50
```

**Acceptance:** Reports success rate, average steps, and average final distance to target.

---

## Stage 6: Open Kaka Adapter Skeleton

**Goal:** Design and implement the Open Kaka robot adapter interface (no real robot).

**Main files:**
- `mini_vla/robot_interface/open_kaka_adapter.py`
- `docs/05_open_kaka_integration.md`

**Acceptance:** Adapter can be imported and called without error.

---

## Future Directions (V1+)

| Direction | Goal |
|-----------|------|
| Action chunk | Predict future N-step action sequence |
| Episode robot dataset | Support real robot episode format |
| Pretrained encoder | Replace small CNN/GRU with ResNet/TinyBERT |
| Real robot rollout | Open Kaka virtual arm + safety loop |
