"""Tests for DatasetRegistry."""

import pytest

from mini_vla.datasets.registry import get_dataset_spec, list_dataset_names


class TestGetDatasetSpec:
    def test_pusht_exists(self):
        spec = get_dataset_spec("pusht")
        assert spec.name == "pusht"
        assert spec.repo_id == "lerobot/pusht"
        assert spec.supports_training is True
        assert spec.image_keys is not None

    def test_aloha_exists(self):
        spec = get_dataset_spec("aloha_sim_transfer_cube")
        assert spec.name == "aloha_sim_transfer_cube"
        assert "observation.images.top" in spec.image_keys

    def test_libero_exists(self):
        spec = get_dataset_spec("libero")
        assert spec.name == "libero"
        assert spec.supports_training is False

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_dataset_spec("nonexistent_dataset")


class TestListDatasetNames:
    def test_contains_expected(self):
        names = list_dataset_names()
        assert "pusht" in names
        assert "aloha_sim_transfer_cube" in names
        assert "libero" in names
