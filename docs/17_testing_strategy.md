# Testing Strategy

## Principle

Tests are categorized by purpose, not by implementation detail.
The goal is to guarantee project correctness without accumulating
redundant shape-check tests for every intermediate function.

## Test Categories

| Category | Purpose | Run by default? |
|----------|---------|:---------------:|
| **Core unit** | Model forward, dataset load, collation, checkpoint I/O | ✅ |
| **Policy behavior** | SingleFrame / History / DeltaAction / ActionChunk loss & prediction | ✅ |
| **Integration smoke** | Training loop, evaluation pipeline, Predictor | ✅ |
| **Audit / Regression** | Stage 7 audit, Stage 8 matrix audit, `audit_stage7_outputs.py` | ✅ |
| **Real-data optional** | Tests downloading real PushT / ALOHA via LeRobot | ❌ (marker `realdata`) |
| **Legacy / Redundant** | Old Toy 2D fine-grained tests that can be merged | ⚠️ keep but deprioritise |

## Tests to Keep

- ``tests/test_config_loader.py`` — core config
- ``tests/test_dataset.py`` — Toy 2D dataset
- ``tests/test_dataset_splits.py`` — episode split integrity
- ``tests/test_model_forward.py``, ``tests/test_model_builder.py`` — model correctness
- ``tests/test_training.py`` — training loop, checkpoint
- ``tests/test_history_wrapper.py``, ``tests/test_delta_action_target.py`` — transforms
- ``tests/test_collate_action_chunk.py`` — collation with extra fields
- ``tests/test_policy_registry.py`` — registry + all policy types
- ``tests/test_policy_high_dim.py`` — ALOHA-scale dimensions
- ``tests/test_predictor_delta_policy.py`` — Predictor delta reconstruction
- ``tests/test_action_chunk_target.py``, ``tests/test_action_chunk_policy.py`` — chunk behaviour
- ``tests/test_action_chunk_eval.py`` — chunk metrics
- ``tests/test_inference_predictor.py`` — Predictor public API
- ``tests/test_shape_meta.py`` — shape_meta utility
- ``tests/test_aloha_shape_meta.py`` — ALOHA key/dim inference (mock)

## Tests That Could Be Merged / Parameterised (Future)

Currently separate files for SingleFrame/History/Delta/Chunk policy tests.
These can be merged into a single parameterised ``test_policies.py``
that covers all registered policies:

```python
@pytest.mark.parametrize("policy_type", [
    "single_frame_bc", "history_bc", "delta_action_bc", "action_chunk_bc",
])
def test_policy_compute_loss_returns_dict(policy_type):
    ...
```

Do this only when the parametrisation is clean — not before.

## What NOT to Test

- Implementation details of internal model submodules
- Exact loss values on random data (flaky)
- Docs rendering or ROADMAP formatting
- CLI help string matching (one smoke test per CLI is enough)
