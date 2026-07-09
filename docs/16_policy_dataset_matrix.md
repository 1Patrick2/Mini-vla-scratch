# Stage 8: Policy × Dataset Matrix

## Motivation

Stage 7 established a strict PushT offline BC benchmark. Stage 8 upgrades
the project from a single-dataset, single-model pipeline into a lightweight
Policy × Dataset Matrix framework for MiniVLA.

## Policy Abstraction

All policies inherit from ``BasePolicy`` and expose a uniform interface:

```python
class BasePolicy(nn.Module):
    policy_type: str

    def compute_loss(self, batch: dict) -> dict:
        """Return {'loss', 'mse', 'mae', ...} used by Trainer."""

    def predict_action(self, batch: dict) -> dict:
        """Return predicted action or action chunk."""

    def get_action_dim(self) -> int:
        """Return the action dimension."""
```

Policies are registered in a central registry and built via ``build_policy(config)``,
allowing config-driven model selection:

```yaml
policy:
  type: single_frame_bc   # or history_bc, delta_action_bc, action_chunk_bc
```

## shape_meta

``shape_meta`` is a declarative config section that resolves state/action
dimensions without hard-wiring PushT-specific values:

```yaml
shape_meta:
  state:
    dim: 2
  action:
    dim: 2
  obs_horizon: 1
  action_horizon: 4
```

## Policy Variants

| Policy | Output | Purpose |
|--------|--------|---------|
| SingleFrame BC | action | current-frame baseline |
| History BC | action | +temporal context |
| DeltaAction BC | delta action | residual action prediction |
| ActionChunk BC | action chunk [B, H, D] | chunked policy smoke |

## Dataset Support Matrix

| Dataset | Status | Policies |
|---------|--------|----------|
| PushT | full benchmark | SingleFrame / History / DeltaAction / ActionChunk smoke |
| ALOHA sim | training smoke | SingleFrame / History / ActionChunk |
| LIBERO | inspect / language feasibility | — |

## ActionChunk BC Design

ActionChunk BC predicts a sequence of ``H`` future actions in one forward pass.
The underlying MiniVLA outputs a flattened vector ``[B, H * action_dim]``,
which the policy reshapes to ``[B, H, action_dim]``.

- ``action_horizon``: via ``shape_meta.action_horizon`` (default 4)
- ``per-step action_dim``: via ``shape_meta.action.dim``
- ``model.action_dim``: ``H * action_dim`` (flattened output)

Loss is MSE over the full chunk. Metrics include ``mae_all`` (full chunk)
and ``mae_first`` (first step, comparable to single-step policies).

The ``ActionChunkTargetWrapper`` generates chunk targets without crossing
episode boundaries — trailing frames with fewer than ``H`` remaining steps
are dropped.

## ALOHA Smoke Setup

ALOHA sim smoke configs are located at:

```
configs/experiments/aloha_sim/single_frame_bc_smoke.yaml
configs/experiments/aloha_sim/history_bc_smoke.yaml
configs/experiments/aloha_sim/action_chunk_bc_smoke.yaml
```

Target: CPU-friendly (1 epoch, 128 max_samples, batch_size=8). Goal is to
run ``dataset → train → eval → report``, not to chase metrics.

Run with real data:
```powershell
$env:RUN_REAL_ROBOT_DATA="1"
python scripts/inspect_shape_meta.py --dataset-name aloha_sim_transfer_cube --repo-id lerobot/aloha_sim_transfer_cube_scripted --loader lerobot --max-samples 128
```

## Verification Commands

```bash
# Default tests (no network)
pytest
ruff check .

# Stage 7 audit
python scripts/audit_stage7_outputs.py

# Stage 8 matrix audit
python scripts/audit_stage8_matrix.py

# Generate matrix summary
python scripts/run_policy_matrix.py --config configs/matrix/stage8_policy_dataset_matrix.yaml --output-dir outputs/matrix
```

## Limitations

- ALOHA is smoke-level support, not a full benchmark.
- LIBERO remains inspect/language feasibility.
- ActionChunk BC is a minimal regression baseline, not full ACT.
- No pretrained VLM/LLM is used yet.
- Diffusion Policy is out of scope for Stage 8.
