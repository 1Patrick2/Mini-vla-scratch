"""Smoke test for training MiniVLA on PushT-like data."""

import numpy as np
import torch

from mini_vla.datasets.pusht_adapter import PushTDatasetAdapter
from mini_vla.models import build_model
from mini_vla.training.checkpoint import save_checkpoint
from mini_vla.training.losses import mse_action_loss
from mini_vla.training.optimizer import create_optimizer

_CONFIG = {
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
    "image_key": "observation.image",
    "state_key": "observation.state",
    "action_key": "action",
    "image_size": 64,
    "instruction": "push the T block to the target",
}


def _make_mock_samples(n=12):
    rng = np.random.RandomState(1)
    return [
        {
            "observation.image": rng.randint(0, 256, (96, 96, 3), dtype=np.uint8),
            "observation.state": rng.randn(2).astype(np.float32),
            "action": rng.randn(2).astype(np.float32),
            "episode_index": i // 4,
            "frame_index": i,
            "timestamp": float(i) * 0.1,
            "next.reward": 0.0,
            "next.done": False,
            "next.success": False,
        }
        for i in range(n)
    ]


class TestPushTTrainingSmoke:
    def test_one_step_no_error(self):
        """Single training step on PushT-like data does not crash."""
        samples = _make_mock_samples()
        ds = PushTDatasetAdapter(samples, **_DATA_CFG)
        from torch.utils.data import DataLoader

        from mini_vla.datasets.collate import collate_toy_2d

        loader = DataLoader(ds, batch_size=4, collate_fn=collate_toy_2d)
        batch = next(iter(loader))

        model = build_model(_CONFIG)
        opt = create_optimizer(model, _CONFIG)

        pred = model(batch)
        loss = mse_action_loss(pred, batch["action"])
        opt.zero_grad()
        loss.backward()
        opt.step()

        assert torch.isfinite(loss)

    def test_trainable_params_update(self):
        """Trainable parameters change after training."""
        samples = _make_mock_samples()
        ds = PushTDatasetAdapter(samples, **_DATA_CFG)
        from torch.utils.data import DataLoader

        from mini_vla.datasets.collate import collate_toy_2d

        loader = DataLoader(ds, batch_size=4, collate_fn=collate_toy_2d)

        model = build_model(_CONFIG)
        opt = create_optimizer(model, _CONFIG)

        before = {
            name: p.clone()
            for name, p in model.action_head.named_parameters()
        }

        for batch in loader:
            pred = model(batch)
            loss = mse_action_loss(pred, batch["action"])
            opt.zero_grad()
            loss.backward()
            opt.step()

        after = {
            name: p.clone()
            for name, p in model.action_head.named_parameters()
        }
        changed = any(
            not torch.equal(before[k], after[k]) for k in before
        )
        assert changed, "Trainable params should update after training"

    def test_checkpoint_save(self, tmp_path):
        """Checkpoint can be saved after training on PushT-like data."""
        samples = _make_mock_samples()
        ds = PushTDatasetAdapter(samples, **_DATA_CFG)
        from torch.utils.data import DataLoader

        from mini_vla.datasets.collate import collate_toy_2d

        loader = DataLoader(ds, batch_size=4, collate_fn=collate_toy_2d)
        model = build_model(_CONFIG)
        opt = create_optimizer(model, _CONFIG)

        for batch in loader:
            pred = model(batch)
            loss = mse_action_loss(pred, batch["action"])
            opt.zero_grad()
            loss.backward()
            opt.step()

        ckpt_path = save_checkpoint(
            tmp_path / "pusht_smoke.pt", model, optimizer=opt,
            epoch=1, metrics={"loss": float(loss.item())},
        )
        assert ckpt_path.exists()
