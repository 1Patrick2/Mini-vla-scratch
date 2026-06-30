# Data Format

The planned episode format is:

```json
{
  "episode_id": "ep_000001",
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

Training batches will expose:

```python
{
    "image": "Tensor[B, 3, H, W]",
    "input_ids": "Tensor[B, T]",
    "state": "Tensor[B, state_dim]",
    "action": "Tensor[B, action_dim]",
}
```
