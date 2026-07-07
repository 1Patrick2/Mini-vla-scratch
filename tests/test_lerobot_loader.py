"""Tests for the generic LeRobot dataset loader."""

from unittest.mock import MagicMock, patch

import pytest

from mini_vla.datasets.lerobot_loader import (
    import_lerobot_dataset,
    load_lerobot_samples,
)


class TestImportLeRobotDataset:
    def test_import_raises_on_missing(self):
        """When neither import path works, raises ImportError with all tried paths."""
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            with pytest.raises(ImportError) as exc:
                import_lerobot_dataset()
            msg = str(exc.value)
            assert "lerobot.datasets.lerobot_dataset" in msg
            assert "lerobot.common.datasets.lerobot_dataset" in msg


class TestLoadLerobotSamples:
    def test_load_with_mock(self):
        """Use a real LeRobotDataset mock to verify load flow."""
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 10
        mock_sample = {
            "observation.image": "dummy",
            "observation.state": [1.0, 2.0],
            "action": [0.5, -0.3],
        }
        # __getitem__ returns the sample for any index
        mock_ds.__getitem__.return_value = mock_sample

        with patch(
            "mini_vla.datasets.lerobot_loader.import_lerobot_dataset",
            return_value=lambda repo_id, root=None: mock_ds,
        ):
            samples = load_lerobot_samples("lerobot/fake", max_samples=3)
            assert len(samples) == 3
            assert samples[0]["observation.image"] == "dummy"

    def test_validate_empty_raises(self):
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 0

        with patch(
            "mini_vla.datasets.lerobot_loader.import_lerobot_dataset",
            return_value=lambda repo_id, root=None: mock_ds,
        ):
            with pytest.raises(ValueError, match="zero samples"):
                load_lerobot_samples("lerobot/fake", validate=True)
