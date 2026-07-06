"""Tests for the prediction visualizer."""


import torch
from PIL import Image

from mini_vla.inference.visualizer import (
    draw_action_arrows,
    save_prediction_visualization,
    tensor_image_to_pil,
)


def _make_dummy_image():
    """Return a dummy uint8 RGB tensor [3, 64, 64]."""
    return torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)


class TestTensorImageToPil:
    def test_converts_to_rgb(self):
        tensor = _make_dummy_image()
        pil_img = tensor_image_to_pil(tensor)
        assert pil_img.mode == "RGB"
        assert pil_img.size == (64, 64)

    def test_float_image(self):
        tensor = torch.rand(3, 64, 64)
        pil_img = tensor_image_to_pil(tensor)
        assert pil_img.mode == "RGB"
        assert pil_img.size == (64, 64)


class TestDrawActionArrows:
    def test_returns_copy(self):
        pil_img = tensor_image_to_pil(_make_dummy_image())
        state = torch.tensor([0.5, 0.5])
        pred = torch.tensor([0.03, -0.01])
        result = draw_action_arrows(pil_img, state, pred)
        assert result.size == pil_img.size
        assert result.mode == "RGB"

    def test_without_gt_action(self):
        pil_img = tensor_image_to_pil(_make_dummy_image())
        state = torch.tensor([0.5, 0.5])
        pred = torch.tensor([0.03, -0.01])
        result = draw_action_arrows(pil_img, state, pred, gt_action=None)
        assert result.size == (64, 64)

    def test_with_gt_action(self):
        pil_img = tensor_image_to_pil(_make_dummy_image())
        state = torch.tensor([0.5, 0.5])
        pred = torch.tensor([0.03, -0.01])
        gt = torch.tensor([0.02, 0.01])
        result = draw_action_arrows(pil_img, state, pred, gt_action=gt)
        assert result.size == (64, 64)


class TestSavePredictionVisualization:
    def test_saves_png(self, tmp_path):
        image = _make_dummy_image()
        state = torch.tensor([0.5, 0.5])
        pred = torch.tensor([0.03, -0.01])
        path = save_prediction_visualization(
            tmp_path / "pred.png", image, state, pred,
        )
        assert path.exists()
        # Can be opened as PNG
        loaded = Image.open(path)
        assert loaded.size == (64, 64)
        assert loaded.mode == "RGB"

    def test_saves_with_gt_action(self, tmp_path):
        image = _make_dummy_image()
        state = torch.tensor([0.5, 0.5])
        pred = torch.tensor([0.03, -0.01])
        gt = torch.tensor([0.02, 0.01])
        path = save_prediction_visualization(
            tmp_path / "pred_gt.png", image, state, pred, gt_action=gt,
        )
        assert path.exists()
        loaded = Image.open(path)
        assert loaded.size == (64, 64)

    def test_saves_without_gt_action(self, tmp_path):
        image = _make_dummy_image()
        state = torch.tensor([0.5, 0.5])
        pred = torch.tensor([0.03, -0.01])
        path = save_prediction_visualization(
            tmp_path / "pred_no_gt.png", image, state, pred, gt_action=None,
        )
        assert path.exists()
        loaded = Image.open(path)
        assert loaded.size == (64, 64)
