#!/usr/bin/env python3
"""Generate synthetic Toy 2D manipulation episodes.

Usage
-----
.. code-block:: bash

    python scripts/generate_toy_data.py --config configs/data/toy_2d.yaml --num-episodes 5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _normalize(x: float, y: float, size: int) -> tuple[float, float]:
    """Convert pixel coordinates to normalized [0, 1]."""
    return (x / size, y / size)


def _denormalize(nx: float, ny: float, size: int) -> tuple[int, int]:
    """Convert normalized [0, 1] to pixel coordinates."""
    return (int(round(nx * size)), int(round(ny * size)))


def draw_frame(
    size: int,
    obj_x: float,
    obj_y: float,
    tgt_x: float,
    tgt_y: float,
    obj_radius: int = 4,
    tgt_radius: int = 4,
) -> Image.Image:
    """Render a single frame with object and target circles.

    Args:
        size: Canvas size in pixels (both width and height).
        obj_x, obj_y: Object position (normalized [0, 1]).
        tgt_x, tgt_y: Target position (normalized [0, 1]).
        obj_radius: Radius of the object circle in pixels.
        tgt_radius: Radius of the target circle in pixels.

    Returns:
        A PIL Image in RGB mode.
    """
    img = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw target (green circle)
    tx, ty = _denormalize(tgt_x, tgt_y, size)
    draw.ellipse(
        [tx - tgt_radius, ty - tgt_radius, tx + tgt_radius, ty + tgt_radius],
        fill=(0, 200, 0),
        outline=(0, 150, 0),
        width=1,
    )

    # Draw object (red circle)
    ox, oy = _denormalize(obj_x, obj_y, size)
    draw.ellipse(
        [ox - obj_radius, oy - obj_radius, ox + obj_radius, oy + obj_radius],
        fill=(220, 40, 40),
        outline=(180, 20, 20),
        width=1,
    )

    return img


def generate_episode(
    episode_id: str,
    instruction: str,
    max_steps: int,
    size: int,
    step_size: float,
    rng: np.random.Generator,
) -> dict:
    """Generate one episode of Toy 2D data.

    Args:
        episode_id: Unique identifier, e.g. ``"ep_000000"``.
        instruction: Language instruction string.
        max_steps: Maximum number of steps per episode.
        size: Canvas size in pixels.
        step_size: Action step size in normalized coordinates.
        rng: NumPy random generator for reproducibility.

    Returns:
        Episode dict with ``episode_id``, ``instruction``, and ``steps``.
    """
    # Random initial object and target positions, avoiding edges (margin 0.1)
    margin = 0.10
    obj_x = rng.uniform(margin, 1.0 - margin)
    obj_y = rng.uniform(margin, 1.0 - margin)
    tgt_x = rng.uniform(margin, 1.0 - margin)
    tgt_y = rng.uniform(margin, 1.0 - margin)

    steps: list[dict] = []
    for step_idx in range(max_steps):
        # Compute action: unit vector toward target * step_size
        dx = tgt_x - obj_x
        dy = tgt_y - obj_y
        dist = np.sqrt(dx * dx + dy * dy)
        if dist > step_size:
            action_x = dx / dist * step_size
            action_y = dy / dist * step_size
        else:
            action_x = dx
            action_y = dy

        # Record step
        steps.append(
            {
                "frame": f"frames/{step_idx:06d}.png",
                "state": [round(obj_x, 6), round(obj_y, 6)],
                "action": [round(action_x, 6), round(action_y, 6)],
                "target": [round(tgt_x, 6), round(tgt_y, 6)],
            }
        )

        # Update object position
        obj_x = np.clip(obj_x + action_x, 0.0, 1.0)
        obj_y = np.clip(obj_y + action_y, 0.0, 1.0)

    return {
        "episode_id": episode_id,
        "instruction": instruction,
        "steps": steps,
    }


def render_episode_frames(episode: dict, size: int, output_dir: Path) -> None:
    """Render all frames for an episode and save as PNG files."""
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Keep a running state for rendering
    obj_x = None
    obj_y = None
    tgt_x = None
    tgt_y = None

    for step in episode["steps"]:
        sx, sy = step["state"]
        tx, ty = step["target"]
        if obj_x is None:
            obj_x, obj_y = sx, sy
            tgt_x, tgt_y = tx, ty
        # Render the *next* frame using the current state
        img = draw_frame(size, obj_x, obj_y, tgt_x, tgt_y)
        frame_path = frames_dir / step["frame"].split("/")[-1]
        img.save(frame_path)

        # Advance state
        ax, ay = step["action"]
        obj_x = np.clip(obj_x + ax, 0.0, 1.0)
        obj_y = np.clip(obj_y + ay, 0.0, 1.0)


def generate_toy_data(
    output_root: str | Path,
    num_episodes: int,
    max_steps: int,
    image_size: int,
    step_size: float = 0.03,
    instruction: str = "move red object to target",
    seed: int = 42,
) -> Path:
    """Generate Toy 2D dataset.

    Args:
        output_root: Root directory for the dataset.
        num_episodes: Number of episodes to generate.
        max_steps: Steps per episode.
        image_size: Canvas size in pixels.
        step_size: Action step size in normalized coordinates.
        instruction: Language instruction for all episodes.
        seed: Random seed for reproducibility.

    Returns:
        The resolved ``output_root`` path.
    """
    rng = np.random.default_rng(seed)
    output_root = Path(output_root)
    episodes_dir = output_root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx in range(num_episodes):
        ep_id = f"ep_{ep_idx:06d}"
        ep_dir = episodes_dir / ep_id
        ep_dir.mkdir(parents=True, exist_ok=True)

        episode = generate_episode(
            episode_id=ep_id,
            instruction=instruction,
            max_steps=max_steps,
            size=image_size,
            step_size=step_size,
            rng=rng,
        )

        # Write episode.json
        json_path = ep_dir / "episode.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(episode, f, indent=2, ensure_ascii=False)

        # Render frames
        render_episode_frames(episode, image_size, ep_dir)

    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Toy 2D data.")
    parser.add_argument(
        "--config",
        default="configs/data/toy_2d.yaml",
        help="Path to data config YAML.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Override number of episodes.",
    )
    args = parser.parse_args()

    # Load config
    import yaml

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    num_episodes = args.num_episodes or data_cfg.get("num_episodes", 1000)
    max_steps = data_cfg.get("max_steps", 16)
    image_size = data_cfg.get("image_size", 64)
    data_root = data_cfg.get("data_root", "data/toy_2d")

    print(f"Generating {num_episodes} Toy 2D episodes ...")
    print(f"  data_root: {data_root}")
    print(f"  max_steps: {max_steps}")
    print(f"  image_size: {image_size}")

    output = generate_toy_data(
        output_root=data_root,
        num_episodes=num_episodes,
        max_steps=max_steps,
        image_size=image_size,
    )

    print(f"Done. Output: {output}")


if __name__ == "__main__":
    main()
