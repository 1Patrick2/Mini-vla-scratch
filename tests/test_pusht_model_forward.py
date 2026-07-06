"""Tests for MiniVLA forward on PushT-like batches."""

import numpy as np
import torch

from mini_vla.datasets.factory import build_dataset
from mini_vla.datasets.pusht_adapter import PushTDatasetAdapter
from mini_vla.models import build_model
from mini_vla.training.losses import mse_action_loss

# Config-compatible with configs/base.yaml (128-dim)
_MODEL_CFG = {
    "model": {
        "action_dim": 2,
        "vision_encoder": {"type": "small_cnn", "output_dim": 128},
        "text_encoder": {"type": "mock_llm", "output_dim": 128, "freeze": True},
        "state_encoder": {"input_dim": 2, "output_dim": 128},
        "fusion": {"type": "concat_mlp", "input_dim": 384, "output_dim": 128},
        "action_head": {"input_dim": 128},
    },
    "train": {"lr": 0.01},
}

_DATA_CFG = {
    "dataset_type": "pusht",
    "image_key": "observation.image",
    "state_key": "observation.state",
    "action_key": "action",
    "image_size": 64,
    "instruction": "push the T block to the target",
}

_ADAPTER_CFG = {
    "image_key": "observation.image",
    "state_key": "observation.state",
    "action_key": "action",
    "image_size": 64,
    "instruction": "push the T block to the target",
}


def _make_mock_push_t_samples(n=8):
    rng = np.random.RandomState(0)
    samples = []
    for i in range(n):
        samples.append({
            "observation.image": rng.randint(0, 256, (96, 96, 3), dtype=np.uint8),
            "observation.state": rng.randn(2).astype(np.float32),
            "action": rng.randn(2).astype(np.float32),
            "episode_index": i // 4,
            "frame_index": i,
            "timestamp": float(i) * 0.1,
            "next.reward": float(i == 3),
            "next.done": i == 3,
            "next.success": i == 3,
        })
    return samples


class TestPushTModelForward:
    def test_build_dataset_returns_adapter(self):
        samples = _make_mock_push_t_samples()
        ds = build_dataset(_DATA_CFG, base_dataset=samples)
        assert isinstance(ds, PushTDatasetAdapter)
        assert len(ds) == 8

    def test_model_forward_on_push_t_batch(self):
        samples = _make_mock_push_t_samples()
        ds = PushTDatasetAdapter(samples, **_ADAPTER_CFG)
        from torch.utils.data import DataLoader

        from mini_vla.datasets.collate import collate_toy_2d

        loader = DataLoader(ds, batch_size=4, collate_fn=collate_toy_2d)
        batch = next(iter(loader))

        model = build_model(_MODEL_CFG)
        pred = model(batch)

        assert pred.shape == (4, 2), f"Expected (4,2), got {pred.shape}"
        assert torch.isfinite(pred).all()

    def test_loss_finite_on_push_t_batch(self):
        samples = _make_mock_push_t_samples()
        ds = PushTDatasetAdapter(samples, **_ADAPTER_CFG)
        from torch.utils.data import DataLoader

        from mini_vla.datasets.collate import collate_toy_2d

        loader = DataLoader(ds, batch_size=4, collate_fn=collate_toy_2d)
        batch = next(iter(loader))

        model = build_model(_MODEL_CFG)
        pred = model(batch)
        loss = mse_action_loss(pred, batch["action"])

        assert loss.ndim == 0
        assert torch.isfinite(loss)
