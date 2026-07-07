"""Tests for DatasetSpec."""

from mini_vla.datasets.spec import DatasetSpec


class TestDatasetSpec:
    def test_create_minimal(self):
        spec = DatasetSpec(name="test", repo_id="lerobot/test")
        assert spec.name == "test"
        assert spec.repo_id == "lerobot/test"

    def test_all_fields(self):
        spec = DatasetSpec(
            name="test",
            repo_id="lerobot/test",
            image_keys=("obs.img",),
            state_keys=("obs.state",),
            action_keys=("action",),
            language_keys=("task",),
            default_instruction="do something",
            supports_training=True,
            supports_evaluation=True,
            notes="test dataset",
        )
        assert spec.default_instruction == "do something"
        assert spec.supports_training is True

    def test_immutable(self):
        spec = DatasetSpec(name="test", repo_id="lerobot/test")
        # Frozen dataclass should prevent attribute assignment
        import pytest
        # Frozen dataclass raises FrozenInstanceError (subclass of AttributeError)
        with pytest.raises(AttributeError):
            spec.name = "changed"  # type: ignore[misc]
