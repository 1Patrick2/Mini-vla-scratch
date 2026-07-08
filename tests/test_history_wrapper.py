"""Tests for HistoryDatasetWrapper."""

import torch

from mini_vla.datasets.transforms import HistoryDatasetWrapper


def _make_frames():
    """Return 5 frames across 2 episodes:
    episode 0: frame 0,1
    episode 1: frame 0,1,2
    """
    return [
        {"episode_index": 0, "frame_index": 0,
         "state": torch.tensor([1.0, 1.0]),
         "action": torch.tensor([10.0, 10.0]),
         "state_raw": torch.tensor([1.0, 1.0]),
         "action_raw": torch.tensor([10.0, 10.0])},
        {"episode_index": 0, "frame_index": 1,
         "state": torch.tensor([2.0, 2.0]),
         "action": torch.tensor([20.0, 20.0]),
         "state_raw": torch.tensor([2.0, 2.0]),
         "action_raw": torch.tensor([20.0, 20.0])},
        {"episode_index": 1, "frame_index": 0,
         "state": torch.tensor([3.0, 3.0]),
         "action": torch.tensor([30.0, 30.0]),
         "state_raw": torch.tensor([3.0, 3.0]),
         "action_raw": torch.tensor([30.0, 30.0])},
        {"episode_index": 1, "frame_index": 1,
         "state": torch.tensor([4.0, 4.0]),
         "action": torch.tensor([40.0, 40.0]),
         "state_raw": torch.tensor([4.0, 4.0]),
         "action_raw": torch.tensor([40.0, 40.0])},
        {"episode_index": 1, "frame_index": 2,
         "state": torch.tensor([5.0, 5.0]),
         "action": torch.tensor([50.0, 50.0]),
         "state_raw": torch.tensor([5.0, 5.0]),
         "action_raw": torch.tensor([50.0, 50.0])},
    ]


class TestHistoryDatasetWrapper:
    def test_first_frame_is_first(self):
        frames = _make_frames()
        wrapper = HistoryDatasetWrapper(frames)
        assert wrapper[0]["is_first_frame"] is True   # ep0 f0
        assert wrapper[2]["is_first_frame"] is True   # ep1 f0

    def test_first_frame_prev_action_zero(self):
        frames = _make_frames()
        wrapper = HistoryDatasetWrapper(frames)
        s = wrapper[0]  # ep0 f0
        assert torch.allclose(s["prev_action_raw"], torch.zeros(2))

    def test_first_frame_prev_state_current(self):
        frames = _make_frames()
        wrapper = HistoryDatasetWrapper(frames)
        s = wrapper[0]
        assert torch.allclose(s["prev_state_raw"], s["state_raw"])

    def test_second_frame_prev_action_matches_first(self):
        frames = _make_frames()
        wrapper = HistoryDatasetWrapper(frames)
        # ep0 f1
        s = wrapper[1]
        assert torch.allclose(s["prev_action_raw"], torch.tensor([10.0, 10.0]))
        assert s["is_first_frame"] is False

    def test_episode_boundary_resets(self):
        frames = _make_frames()
        wrapper = HistoryDatasetWrapper(frames)
        s0 = wrapper[1]  # ep0 f1
        s1 = wrapper[2]  # ep1 f0
        assert s0["action_raw"][0] == 20.0
        assert s1["is_first_frame"] is True
        assert torch.allclose(s1["prev_action_raw"], torch.zeros(2))

    def test_concat_to_state(self):
        frames = _make_frames()
        wrapper = HistoryDatasetWrapper(frames, concat_to_state=True)
        s = wrapper[0]  # ep0 f0: state=[1,1], prev_state=[1,1], prev_action=[0,0]
        assert s["state"].shape == (6,)
        assert torch.allclose(s["state"][:2], torch.tensor([1.0, 1.0]))
        assert torch.allclose(s["state"][2:4], torch.tensor([1.0, 1.0]))
        assert torch.allclose(s["state"][4:], torch.zeros(2))

    def test_concat_non_first_frame(self):
        frames = _make_frames()
        wrapper = HistoryDatasetWrapper(frames, concat_to_state=True)
        s = wrapper[1]  # ep0 f1: state=[2,2], prev_state=[1,1], prev_action=[10,10]
        assert s["state"].shape == (6,)
        assert torch.allclose(s["state"][:2], torch.tensor([2.0, 2.0]))
        assert torch.allclose(s["state"][2:4], torch.tensor([1.0, 1.0]))
        assert torch.allclose(s["state"][4:], torch.tensor([10.0, 10.0]))

    def test_normalized_fields_correct(self):
        """When state_normalized/action_normalized exist, prev_* use them."""
        frames = [
            {"episode_index": 0, "frame_index": 0,
             "state": torch.tensor([1.0, 1.0]),
             "action": torch.tensor([10.0, 10.0]),
             "state_normalized": torch.tensor([-0.5, -0.5]),
             "action_normalized": torch.tensor([-1.0, -1.0]),
             "state_raw": torch.tensor([1.0, 1.0]),
             "action_raw": torch.tensor([10.0, 10.0])},
            {"episode_index": 0, "frame_index": 1,
             "state": torch.tensor([2.0, 2.0]),
             "action": torch.tensor([20.0, 20.0]),
             "state_normalized": torch.tensor([0.5, 0.5]),
             "action_normalized": torch.tensor([1.0, 1.0]),
             "state_raw": torch.tensor([2.0, 2.0]),
             "action_raw": torch.tensor([20.0, 20.0])},
        ]
        wrapper = HistoryDatasetWrapper(frames)
        s = wrapper[1]  # ep0 f1
        # prev fields should use normalized space
        assert torch.allclose(s["prev_state"], torch.tensor([-0.5, -0.5]))
        assert torch.allclose(s["prev_action"], torch.tensor([-1.0, -1.0]))
        # raw fields preserved
        assert torch.allclose(s["prev_state_raw"], torch.tensor([1.0, 1.0]))
        assert torch.allclose(s["prev_action_raw"], torch.tensor([10.0, 10.0]))

    def test_repeat_access_no_mutation(self):
        """Accessing the same index multiple times should not change state shape."""
        frames = _make_frames()
        wrapper = HistoryDatasetWrapper(frames, concat_to_state=True)
        s1 = wrapper[1]
        s2 = wrapper[1]
        s3 = wrapper[1]
        assert s1["state"].shape == (6,), "First access should be 6-dim"
        assert s2["state"].shape == (6,), "Second access should still be 6-dim"
        assert s3["state"].shape == (6,), "Third access should still be 6-dim"
        # Values should be identical across accesses
        assert torch.allclose(s1["state"], s2["state"])
        assert torch.allclose(s2["state"], s3["state"])
