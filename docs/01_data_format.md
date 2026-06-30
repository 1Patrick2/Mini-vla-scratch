# Data Format

## Toy 2D Dataset Structure

Each episode is a self-contained directory with its own metadata and frame images:

```
data/toy_2d/
└── episodes/
    ├── ep_000000/
    │   ├── frames/
    │   │   ├── 000000.png
    │   │   ├── 000001.png
    │   │   └── ...
    │   └── episode.json
    ├── ep_000001/
    │   ├── frames/
    │   └── episode.json
    └── ...
```

## Episode JSON

Each `episode.json` describes one episode with a fixed language instruction and a sequence of steps.

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
    },
    {
      "frame": "frames/000001.png",
      "state": [0.28, 0.39],
      "action": [0.02, 0.00],
      "target": [0.80, 0.20]
    }
  ]
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | string | Unique episode identifier |
| `instruction` | string | Natural language instruction for the entire episode |
| `steps` | array | List of per-timestep samples |
| `steps[].frame` | string | Relative path to the frame image (PNG) |
| `steps[].state` | array[2] | Current position of the red object `[x, y]`, normalized to [0, 1] |
| `steps[].action` | array[2] | Expert action to move the object `[dx, dy]` |
| `steps[].target` | array[2] | Target position `[x, y]`, normalized to [0, 1] |

## Frame Images

- Format: RGB PNG, 64x64 pixels
- Background: white or light gray
- Red circle: current object position (radius ~4 px)
- Green circle: target position (radius ~4 px)
- File naming: 6-digit zero-padded index (`000000.png`, `000001.png`, ...)

## Instructions

Instructions are short fixed templates describing the goal direction:

| Instruction | Meaning |
|-------------|---------|
| `"move red object to target"` | Move directly toward target |
| `"move red object left"` | Move left (negative dx) |
| `"move red object right"` | Move right (positive dx) |
| `"move red object up"` | Move up (negative dy) |
| `"move red object down"` | Move down (positive dy) |

## Dataset Sample Format

Each dataset sample returned by the `Toy2DDataset`:

```python
sample = {
    "image":       Tensor[3, H, W]       # uint8 or float32 normalized RGB
    "input_ids":   Tensor[T]             # tokenized instruction
    "state":       Tensor[state_dim]     # object position [x, y]
    "action":      Tensor[action_dim]    # expert action [dx, dy]
    "instruction": str                   # raw instruction text (optional, for debugging)
}
```

## DataLoader Batch Format

After collation, each batch:

```python
batch = {
    "image":     Tensor[B, 3, H, W]     # batched images
    "input_ids": Tensor[B, T]           # batched token sequences
    "state":     Tensor[B, state_dim]   # batched states
    "action":    Tensor[B, action_dim]  # batched actions
}
```

## Data Splits

Toy data generation should produce separate splits:

```
data/toy_2d/
└── episodes/
    ├── train/       # ~80% of episodes
    └── val/         # ~20% of episodes
```

This split can be handled either at generation time or in the dataset loader.

## Optional Global Manifest

A global `episodes.json` at the dataset root is optional and not required for Stage 1.
If present, it indexes all episodes:

```json
[
  {"episode_id": "ep_000000", "path": "episodes/train/ep_000000/episode.json"},
  {"episode_id": "ep_000001", "path": "episodes/val/ep_000001/episode.json"}
]
```
