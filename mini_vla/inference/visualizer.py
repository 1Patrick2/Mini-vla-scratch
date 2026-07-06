"""Prediction visualizer for MiniVLA inference.

Draws predicted and ground-truth action arrows on the input frame
using PIL, without external dependencies beyond Pillow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageDraw


def tensor_image_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a PyTorch tensor image to a PIL ``RGB`` image.

    Args:
        image: Tensor[C, H, W] in ``[0, 255]`` uint8 or float.

    Returns:
        PIL ``Image`` in RGB mode.
    """
    arr = image.cpu().detach()
    if arr.is_floating_point():
        arr = arr.clamp(0, 1).mul(255).byte()
    # (C, H, W) → (H, W, C)
    arr = arr.permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode="RGB")


def _pixel_coords(
    state: torch.Tensor,
    action: torch.Tensor,
    image_size: int,
    arrow_scale: float,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Convert normalized state+action to pixel start/end coordinates."""
    sx = int(state[0].item() * image_size)
    sy = int(state[1].item() * image_size)
    ex = int(sx + action[0].item() * image_size * arrow_scale)
    ey = int(sy + action[1].item() * image_size * arrow_scale)
    return (sx, sy), (ex, ey)


def draw_action_arrows(
    image: Image.Image,
    state: torch.Tensor,
    pred_action: torch.Tensor,
    gt_action: Optional[torch.Tensor] = None,
    arrow_scale: float = 4.0,
) -> Image.Image:
    """Draw action arrows on a PIL image.

    Args:
        image: PIL ``RGB`` image to draw on.
        state: Tensor[2] — object position ``[x, y]`` in ``[0, 1]``.
        pred_action: Tensor[2] — predicted action ``[dx, dy]``.
        gt_action: Optional Tensor[2] — ground-truth action for comparison.
        arrow_scale: Multiplier to make small Toy2D actions visible.

    Returns:
        New PIL ``RGB`` image with arrows drawn (input is not modified).
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    def _draw_arrow(start, end, color, width=2):
        draw.line([start, end], fill=color, width=width)
        # Arrowhead: small dashes at the tip
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length < 2:
            return
        ux, uy = dx / length, dy / length
        head_len = min(8.0, length * 0.3)
        for sign in (-1, 1):
            hx = int(end[0] - head_len * (ux * 0.866 + sign * uy * 0.5))
            hy = int(end[1] - head_len * (uy * 0.866 - sign * ux * 0.5))
            draw.line([end, (hx, hy)], fill=color, width=width)

    # Predicted action arrow (blue)
    start, end = _pixel_coords(state, pred_action, w, arrow_scale)
    _draw_arrow(start, end, "blue")

    # Ground-truth action arrow (green)
    if gt_action is not None:
        start, end = _pixel_coords(state, gt_action, w, arrow_scale)
        _draw_arrow(start, end, "green")

    return img


def save_prediction_visualization(
    output_path: str | Path,
    image: torch.Tensor,
    state: torch.Tensor,
    pred_action: torch.Tensor,
    gt_action: Optional[torch.Tensor] = None,
    arrow_scale: float = 4.0,
) -> Path:
    """Save a prediction visualization to disk.

    Args:
        output_path: Where to write the ``.png`` file.
        image: Tensor[C, H, W] — the input frame.
        state: Tensor[2] — object position.
        pred_action: Tensor[2] — predicted action.
        gt_action: Optional Tensor[2] — ground-truth action.
        arrow_scale: Arrow length multiplier.

    Returns:
        The resolved ``Path`` of the saved file.
    """
    pil_img = tensor_image_to_pil(image)
    vis_img = draw_action_arrows(
        pil_img, state, pred_action, gt_action, arrow_scale,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vis_img.save(path)
    return path


__all__ = [
    "draw_action_arrows",
    "save_prediction_visualization",
    "tensor_image_to_pil",
]
