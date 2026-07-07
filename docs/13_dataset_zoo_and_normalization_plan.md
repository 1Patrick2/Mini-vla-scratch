# Dataset Zoo and Normalization Plan

## Motivation

Stage 5 proved MiniVLA can train and evaluate on real PushT data with vision.
Stage 6 extends this to a **multi-dataset framework**:

- **Dataset Zoo**: support multiple LeRobot-style datasets through a shared
  specification and adapter system, without per-dataset hard-coding.
- **Action Normalization**: normalise state & action values to zero-mean
  unit-variance, fixing the large-loss / unstable-training issue caused by
  PushT's pixel-level action scale (~0–255).

## Supported Dataset Tiers

### Tier 1 — Full support (train + eval)
| Dataset | ID | Status |
|---------|----|--------|
| PushT | `lerobot/pusht` | ✅ Stage 5 |
| PushT (normalised) | `lerobot/pusht` | 📅 Stage 6-E |

### Tier 2 — Inspect & adapter smoke only
| Dataset | ID | Status |
|---------|----|--------|
| ALOHA sim transfer cube | `lerobot/aloha_sim_transfer_cube_scripted` | 📅 Stage 6-F |
| LIBERO | `lerobot/libero` | 📅 Stage 6-F |

### Tier 3 — Future exploration
| Dataset | ID | Notes |
|---------|----|-------|
| DROID | `lerobot/droid_1.0.1` | Multi-camera, heavy download |
| BridgeData / Open X-Embodiment | Various | Very large scale |

## DatasetSpec Design

A dataclass that declares a dataset's schema:

```python
@dataclass
class DatasetSpec:
    name: str
    repo_id: str
    loader: str = "lerobot"
    image_keys: tuple[str, ...] = ()
    state_keys: tuple[str, ...] = ()
    action_keys: tuple[str, ...] = ()
    language_keys: tuple[str, ...] = ()
    episode_index_key: str = "episode_index"
    frame_index_key: str = "frame_index"
    default_instruction: str | None = None
    supports_training: bool = False
    supports_evaluation: bool = False
    notes: str = ""
```

Every component (inspect, adapter, train, eval) reads from the spec instead
of hard-coding key names.

## DatasetRegistry Design

A module that maps dataset names to `DatasetSpec` objects:

```python
def get_dataset_spec(name: str) -> DatasetSpec: ...
```

Initial registry includes PushT, ALOHA sim, and LIBERO specs.

## Generic Adapter Design

`BaseRobotDatasetAdapter` reads samples from any LeRobotDataset using the
spec's key lists.  It tries each candidate key in order, supports flat and
nested dotted keys, and optionally applies a normalizer.

## Action Normalization Design

### NormalizationStats

```python
@dataclass
class NormalizationStats:
    state_mean: torch.Tensor
    state_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor
    eps: float = 1e-6
```

### ActionNormalizer

```python
normalize_state(state)  → (state - mean) / (std + eps)
normalize_action(action) → (action - mean) / (std + eps)
denormalize_action(action_norm) → action_norm * (std + eps) + mean
```

Stats are computed from a sample of the dataset and saved as JSON, then
loaded at training/evaluation time.

## CLI Design

General-purpose inspect CLI replacing per-dataset scripts:

```bash
python scripts/inspect_robot_dataset.py \
  --dataset-name pusht \
  --repo-id lerobot/pusht \
  --loader lerobot \
  --max-samples 8 \
  --output outputs/dataset_reports/pusht_report.json
```

Stats computation CLI:

```bash
python scripts/compute_dataset_stats.py \
  --dataset-name pusht \
  --repo-id lerobot/pusht \
  --loader lerobot \
  --max-samples 256 \
  --output outputs/dataset_reports/pusht_stats.json
```

## Test Strategy

| Test type | Marker | Default | Runs on CI |
|-----------|--------|---------|------------|
| Unit / mock | — | ✅ Yes | ✅ Yes |
| Realdata | `@pytest.mark.realdata` | ❌ No | ❌ No |

Default tests use mock data, no network, no real downloads.

## Realdata Strategy

Run real data tests locally:

```powershell
$env:RUN_REAL_DATASET_ZOO="1"
python -m pytest tests/test_inspect_robot_dataset_realdata.py -m realdata
```

## Acceptance Criteria (Stage 6)

- [x] README/SETUP/PLAN/ROADMAP synced to Stage 6
- [x] `requirements-robot.txt` created (optional)
- [x] `DatasetSpec` + `Registry` defined and tested
- [x] `key_utils.py` (flat + nested dotted key support)
- [x] Generic `lerobot_loader.py` (no longer PushT-specific)
- [x] Generic `inspect_robot_dataset.py` works with mock and real PushT
- [x] `BaseRobotDatasetAdapter` implemented
- [x] `factory.py` supports `dataset_type=robot_dataset`
- [x] Normalization stats compute/test works
- [x] Normalized PushT train/eval works
- [x] Default pytest does not require network
- [x] `ruff check .` passes
- [ ] ALOHA sim inspect passes or gives clear diagnostic
- [ ] LIBERO inspect passes or gives clear diagnostic
